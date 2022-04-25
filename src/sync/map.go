// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sync

import (
	"sync/atomic"
	"unsafe"
)

// Map is like a Go map[interface{}]interface{} but is safe for concurrent use
// by multiple goroutines without additional locking or coordination.
// Loads, stores, and deletes run in amortized constant time.
//
// The Map type is specialized. Most code should use a plain Go map instead,
// with separate locking or coordination, for better type safety and to make it
// easier to maintain other invariants along with the map content.
//
// The Map type is optimized for two common use cases: (1) when the entry for a given
// key is only ever written once but read many times, as in caches that only grow,
// or (2) when multiple goroutines read, write, and overwrite entries for disjoint
// sets of keys. In these two cases, use of a Map may significantly reduce lock
// contention compared to a Go map paired with a separate Mutex or RWMutex.
//
// The zero Map is empty and ready for use. A Map must not be copied after first use.
type Map struct {
	mu Mutex

	// read contains the portion of the map's contents that are safe for
	// concurrent access (with or without mu held).
	//
	// The read field itself is always safe to load, but must only be stored with
	// mu held.
	//
	// Entries stored in read may be updated concurrently without mu, but updating
	// a previously-expunged entry requires that the entry be copied to the dirty
	// map and unexpunged with mu held.
	// 基本上你可以把它看成一个安全的只读的map
	// 但如果更新，则需要加锁保护。对于read存储的entry字段，可能会被并发的CAS更新。但是如果要更新一个之前已经删除的entry
	// 则需要先将其状态从unexpunged改为nil,再复制到dirty中，然后在更新
	read atomic.Value // readOnly

	// dirty contains the portion of the map's contents that require mu to be
	// held. To ensure that the dirty map can be promoted to the read map quickly,
	// it also includes all of the non-expunged entries in the read map.
	//
	// Expunged entries are not stored in the dirty map. An expunged entry in the
	// clean map must be unexpunged and added to the dirty map before a new value
	// can be stored to it.
	//
	// If the dirty map is nil, the next write to the map will initialize it by
	// making a shallow copy of the clean map, omitting stale entries.
	// 包括所有在read字段中但未被expunged（删除）的元素以及新加的元素
	// 如果dirty为nil，那么下次写入时会新建一个新的dirty，它是read的一个复制，但是剔除了其中已经被删除的key
	dirty map[any]*entry

	// misses counts the number of loads since the read map was last updated that
	// needed to lock mu to determine whether the key was present.
	//
	// Once enough misses have occurred to cover the cost of copying the dirty
	// map, the dirty map will be promoted to the read map (in the unamended
	// state) and the next store to the map will make a new dirty copy.
	// 记录从read中读取miss的次数，一旦miss数和dirty长度一样，就会把dirty提升为read，并把dirty置空
	// misses 字段用来统计 read 被穿透的次数（被穿透指需要读 dirty 的情况）
	misses int
}

// readOnly is an immutable struct stored atomically in the Map.read field.
type readOnly struct {
	m map[any]*entry
	// 当dirty中包含read没有的数据时为true，比如新增一条数据
	amended bool // true if the dirty map contains some key not in m.
}

// expunged is an arbitrary pointer that marks entries which have been deleted
// from the dirty map.
// expunged是用来标识此项已经删掉的指针
// 当map中的一个项目被删除了，只是把它的值标记为expunged，以后才有机会真正删除此项
var expunged = unsafe.Pointer(new(any))

// An entry is a slot in the map corresponding to a particular key.
type entry struct {
	// p points to the interface{} value stored for the entry.
	//
	// If p == nil, the entry has been deleted, and either m.dirty == nil or
	// m.dirty[key] is e.
	// 当p == nil，说明这个kv已经被删除，并且 m.dirty == nil或者 m.dirty[key]指向该entry
	//
	// If p == expunged, the entry has been deleted, m.dirty != nil, and the entry
	// is missing from m.dirty.
	// 当 p == expunged，说明这个kv已经被删除，并且 m.dirty != nil且 m.dirty没有这个key

	//  nil和expunged都代表元素被删除了，只不过expunged比较特殊，
	// 如果被删除的元素是expunged,代表它只存在于readonly之中，不存在于dirty中。
	// 这样如果重新设置这个key的话，需要往dirty增加key

	// Otherwise, the entry is valid and recorded in m.read.m[key] and, if m.dirty
	// != nil, in m.dirty[key].
	// 其他情况，p指向一个正常值，表示实际interface{}的地址。并且记录在m.read.m[key]中。如果这时m.dirty不为nil，那么
	// 也会被记录在m.dirty中，两者实际上值得同一个值

	// An entry can be deleted by atomic replacement with nil: when m.dirty is
	// next created, it will atomically replace nil with expunged and leave
	// m.dirty[key] unset.
	// 当删除key时，并不直接删除。一个entry可以通过原子地CAS设置p为nil(假设删除)。
	// 如果之后创建m.dirty,p又会被原子地设置为expunged，且不会复制到dirty中

	//
	// An entry's associated value can be updated by atomic replacement, provided
	// p != expunged. If p == expunged, an entry's associated value can be updated
	// only after first setting m.dirty[key] = e so that lookups using the dirty
	// map find the entry.
	// 如果p不为expunged，和entry相关联的这个value可以被原子的更新；
	// 如果p==expunged，那么仅当它初次被设置到dirty之后，才可以被更新

	p unsafe.Pointer // *interface{}
}

func newEntry(i any) *entry {
	return &entry{p: unsafe.Pointer(&i)}
}

// Load returns the value stored in the map for a key, or nil if no
// value is present.
// The ok result indicates whether value was found in the map.
func (m *Map) Load(key any) (value any, ok bool) {
	read, _ := m.read.Load().(readOnly)
	e, ok := read.m[key]
	// 如果不存在并且dirty有read中没有的key
	if !ok && read.amended {
		m.mu.Lock()
		// Avoid reporting a spurious miss if m.dirty got promoted while we were
		// blocked on m.mu. (If further loads of the same key will not miss, it's
		// not worth copying the dirty map for this key.)
		// double check.避免在上锁的过程中，dirty被提升为read
		read, _ = m.read.Load().(readOnly)
		e, ok = read.m[key]
		if !ok && read.amended {
			e, ok = m.dirty[key]
			// Regardless of whether the entry was present, record a miss: this key
			// will take the slow path until the dirty map is promoted to the read
			// map.
			// 不管dirty中存不存在，miss数都加1;可能会提升
			m.missLocked()
		}
		m.mu.Unlock()
	}
	// 如果 在dirty中都没有找到，说明就没有这个key
	if !ok {
		return nil, false
	}
	//返回读取的对象，e既可能是从read中获得的(fast path)，也可能是从dirty中获得的(slow path)
	return e.load()
}

func (e *entry) load() (value any, ok bool) {
	p := atomic.LoadPointer(&e.p)
	if p == nil || p == expunged {
		return nil, false
	}
	return *(*any)(p), true
}

// Store sets the value for a key.
func (m *Map) Store(key, value any) {
	read, _ := m.read.Load().(readOnly)
	// 如果在read中存在说明是更新 cas更新成功就行
	if e, ok := read.m[key]; ok && e.tryStore(&value) {
		return
	}
	// read中不存在 或者 cas更新是失败，需要加锁去dirty看看了
	m.mu.Lock()
	read, _ = m.read.Load().(readOnly)
	if e, ok := read.m[key]; ok { // 双检查，看看read是否已经存在了
		if e.unexpungeLocked() {
			// The entry was previously expunged, which implies that there is a
			// non-nil dirty map and this entry is not in it.
			// 如果此时read存在该key了，但p==expunged,则说明m.dirty!=nil且m.dirty不存在该key值
			// a、将p的状态由expunged更改为nil  b、将key插入m.dirty

			// 此entry先前已经被删除了，通过将它的值设置为nil，标记为unexpunged
			m.dirty[key] = e
		}
		// 更新
		e.storeLocked(&value)
	} else if e, ok := m.dirty[key]; ok { // 如果dirty中有
		e.storeLocked(&value) // 直接更新(此时read是没有这个key的)
	} else {
		// 否则，说明是新key(read dirty都没有)，则
		// a' m.dirtyLocked() 如果dirty为空，则需要创建dirty，并从read中复制没有被删除的元素到dirty
		// b' 更新read.amended字段，标识dirty中有read没有的key
		// c' 将kv写入dirty，read不变

		if !read.amended {
			// We're adding the first new key to the dirty map.
			// Make sure it is allocated and mark the read-only map as incomplete.
			m.dirtyLocked()
			m.read.Store(readOnly{m: read.m, amended: true})
		}
		m.dirty[key] = newEntry(value) //将新值写入到dirty
	}
	m.mu.Unlock()
}

// tryStore stores a value if the entry has not been expunged.
//
// If the entry is expunged, tryStore returns false and leaves the entry
// unchanged.
func (e *entry) tryStore(i *any) bool {
	for {
		p := atomic.LoadPointer(&e.p)
		if p == expunged {
			return false
		}
		if atomic.CompareAndSwapPointer(&e.p, p, unsafe.Pointer(i)) {
			return true
		}
	}
}

// unexpungeLocked ensures that the entry is not marked as expunged.
// unexpungellocked 确保entry没有被标记为已删除。
// If the entry was previously expunged, it must be added to the dirty map
// before m.mu is unlocked.

// 如果entry先前已经被清除过了，那么在mutex解锁之前，它一定要被加入到dirty中
func (e *entry) unexpungeLocked() (wasExpunged bool) {
	return atomic.CompareAndSwapPointer(&e.p, expunged, nil)
}

// storeLocked unconditionally stores a value to the entry.
//
// The entry must be known not to be expunged.
func (e *entry) storeLocked(i *any) {
	atomic.StorePointer(&e.p, unsafe.Pointer(i))
}

// LoadOrStore returns the existing value for the key if present.
// Otherwise, it stores and returns the given value.
// The loaded result is true if the value was loaded, false if stored.
func (m *Map) LoadOrStore(key, value any) (actual any, loaded bool) {
	// Avoid locking if it's a clean hit.
	read, _ := m.read.Load().(readOnly)
	if e, ok := read.m[key]; ok {
		actual, loaded, ok := e.tryLoadOrStore(value)
		if ok {
			return actual, loaded
		}
	}

	m.mu.Lock()
	read, _ = m.read.Load().(readOnly)
	if e, ok := read.m[key]; ok {
		if e.unexpungeLocked() {
			m.dirty[key] = e
		}
		actual, loaded, _ = e.tryLoadOrStore(value)
	} else if e, ok := m.dirty[key]; ok {
		actual, loaded, _ = e.tryLoadOrStore(value)
		m.missLocked()
	} else {
		if !read.amended {
			// We're adding the first new key to the dirty map.
			// Make sure it is allocated and mark the read-only map as incomplete.
			m.dirtyLocked()
			m.read.Store(readOnly{m: read.m, amended: true})
		}
		m.dirty[key] = newEntry(value)
		actual, loaded = value, false
	}
	m.mu.Unlock()

	return actual, loaded
}

// tryLoadOrStore atomically loads or stores a value if the entry is not
// expunged.
//
// If the entry is expunged, tryLoadOrStore leaves the entry unchanged and
// returns with ok==false.
func (e *entry) tryLoadOrStore(i any) (actual any, loaded, ok bool) {
	p := atomic.LoadPointer(&e.p)
	if p == expunged {
		return nil, false, false
	}
	if p != nil {
		return *(*any)(p), true, true
	}

	// Copy the interface after the first load to make this method more amenable
	// to escape analysis: if we hit the "load" path or the entry is expunged, we
	// shouldn't bother heap-allocating.
	ic := i
	for {
		if atomic.CompareAndSwapPointer(&e.p, nil, unsafe.Pointer(&ic)) {
			return i, false, true
		}
		p = atomic.LoadPointer(&e.p)
		if p == expunged {
			return nil, false, false
		}
		if p != nil {
			return *(*any)(p), true, true
		}
	}
}

// LoadAndDelete deletes the value for a key, returning the previous value if any.
// The loaded result reports whether the key was present.
func (m *Map) LoadAndDelete(key any) (value any, loaded bool) {
	read, _ := m.read.Load().(readOnly)
	e, ok := read.m[key]
	// 如果read不存在这个key 且 dirty中包含read没有的数据时
	if !ok && read.amended {
		m.mu.Lock()
		read, _ = m.read.Load().(readOnly)
		e, ok = read.m[key]
		if !ok && read.amended {
			e, ok = m.dirty[key]
			delete(m.dirty, key) // 从dirty中删除
			// Regardless of whether the entry was present, record a miss: this key
			// will take the slow path until the dirty map is promoted to the read
			// map.
			m.missLocked()
		}
		m.mu.Unlock()
	}
	// 如果在read或dirty中有这个key，value设置为nil
	if ok {
		return e.delete()
	}
	return nil, false
}

// Delete deletes the value for a key.
func (m *Map) Delete(key any) {
	m.LoadAndDelete(key)
}

func (e *entry) delete() (value any, ok bool) {
	for {
		p := atomic.LoadPointer(&e.p)
		if p == nil || p == expunged {
			return nil, false
		}
		if atomic.CompareAndSwapPointer(&e.p, p, nil) {
			return *(*any)(p), true
		}
	}
}

/*
如果key同时存在read和dirty中，删除只是做个标记，将p=nil，而如果仅在dirty中含有这个key，会直接删除这个key。
原因在于，若两者都存在这个key，仅做标记删除，可以在下一次查找这个key时，命中read，提升效率。若只有在dirty存在，read
起不到缓存的作用，直接删除

值得一提的是，delete通过将值变成nil只能保证key对应的val能够最终被回收，而key本身需要在missLocked中将key从dirty
中进行删除，否则将导致sync.Map存储的key永远无法被垃圾回收

*/

// Range calls f sequentially for each key and value present in the map.
// If f returns false, range stops the iteration.
//
// Range does not necessarily correspond to any consistent snapshot of the Map's
// contents: no key will be visited more than once, but if the value for any key
// is stored or deleted concurrently (including by f), Range may reflect any
// mapping for that key from any point during the Range call. Range does not
// block other methods on the receiver; even f itself may call any method on m.
//
// Range may be O(N) with the number of elements in the map even if f returns
// false after a constant number of calls.
func (m *Map) Range(f func(key, value any) bool) {
	// We need to be able to iterate over all of the keys that were already
	// present at the start of the call to Range.
	// If read.amended is false, then read.m satisfies that property without
	// requiring us to hold m.mu for a long time.
	read, _ := m.read.Load().(readOnly)
	if read.amended { // 如果dirty有read不包含的key，则直接 将dirty提升为read
		// m.dirty contains keys not in read.m. Fortunately, Range is already O(N)
		// (assuming the caller does not break out early), so a call to Range
		// amortizes an entire copy of the map: we can promote the dirty copy
		// immediately!
		m.mu.Lock()
		read, _ = m.read.Load().(readOnly)
		if read.amended {
			read = readOnly{m: m.dirty}
			m.read.Store(read)
			m.dirty = nil
			m.misses = 0
		}
		m.mu.Unlock()
	}

	for k, e := range read.m {
		v, ok := e.load()
		if !ok {
			continue
		}
		if !f(k, v) {
			break
		}
	}
}

func (m *Map) missLocked() {
	// 未命中次数+1
	m.misses++
	// 如果未命中数小于 dirty的长度 就返回
	if m.misses < len(m.dirty) {
		return
	}
	// 如果大于或等于，则将 dirty 提升为
	m.read.Store(readOnly{m: m.dirty})
	m.dirty = nil
	m.misses = 0
}

func (m *Map) dirtyLocked() {
	if m.dirty != nil {
		return
	}
	// 如果dirty为空
	read, _ := m.read.Load().(readOnly)
	m.dirty = make(map[any]*entry, len(read.m))
	// 将read中 没有标记为expunged的 kv赋值到dirty
	for k, e := range read.m {
		if !e.tryExpungeLocked() {
			m.dirty[k] = e
		}
	}
}

// 在新建dirty的时候被调用，会将已删除的p从nil改为expunged，这个entry就不会写入到dirty了
func (e *entry) tryExpungeLocked() (isExpunged bool) {
	p := atomic.LoadPointer(&e.p)
	for p == nil {
		if atomic.CompareAndSwapPointer(&e.p, nil, expunged) {
			return true
		}
		p = atomic.LoadPointer(&e.p)
	}
	return p == expunged
}
