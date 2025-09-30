# Data Structures and Algorithms

In the previous chapter, we created classes for the `stack` and `queue` abstract datatypes. In this chapter, we will continue with the `list` and `set` abstract datatypes.

Similar to the stack and queue of the previous chapter, a list is an ordered structure with a beginning and an end. However, it is possible to add and remove values at any position in the list. It is also possible to iterate through the list.

A set, on the other hand, is an unordered structure of values. The only thing we can say about a set is whether a certain value is present. We cannot say that a value has any position in relation to any other value.

In this chapter, we will look at the following topics:

*   We will start with a rather simple and ineffective version of the list and set classes. We will also look into basic algorithms for searching and sorting.
*   Then we will continue by creating more advanced versions of the list and set classes, and look into more advanced searching and sorting algorithms. We will also introduce new concepts such as templates, operator overloading, exceptions, and reference overloading.

We will also look into the searching algorithms linear search, which works on every sequence, ordered and unordered, but is rather ineffective, and binary search, which is more effective but only works on ordered sequences.

Finally, we will study the rather simple sorting algorithms, insert sort, select sort, and bubble sort, as well as the more advanced and more effective merge sort and quick sort algorithms.

# The List class

The `LinkedList` class is a more complicated abstract data type than the stack and the queue. It is possible to add and remove values at any location in the list. It is also possible to iterate through the list.

# The Cell class

The cell of this section is an extension of the cell of the `stack` and `queue` sections. Similar to them, it holds a value and a pointer to the next cell. However, this version also holds a pointer to the previous cell, which makes the list of this section a double-linked list.

Note that the constructor is `private`, which means that the cell object can be created by its own methods only. However, there is a way to circumvent that limitation. We can define a class or a function to be a friend of `LinkedList`. In this way, we define `LinkedList` as a friend of `Cell`. This means that `LinkedList` has access to all private and protected members of `Cell`, including the constructor, and can thereby create `Cell` objects.

**Cell.h:**

```cpp
class Cell { 
  private: 
    Cell(double value, Cell *previous, Cell *next); 
    friend class LinkedList; 

  public: 
    double getValue() const { return m_value; } 
    void setValue(double value) { m_value = value; } 

    Cell *getPrevious() const { return m_previous; } 
    void setPrevious(Cell *previous) { m_previous = previous; } 

    Cell *getNext() const { return m_next; } 
    void setNext(Cell *getNext) { m_next = getNext; } 

  private: 
    double m_value; 
    Cell *m_previous, *m_next; 
}; 
```

**Cell.cpp:**

```cpp
#include "Cell.h" 

Cell::Cell(double value, Cell *previous, Cell *next) 
 :m_value(value), 
  m_previous(previous), 
  m_next(next) { 
  // Empty. 
} 
```

# The Iterator class

When going through a list, we need an iterator, which is initialized to the beginning of the list and step-wise moves to its end. Similar to the preceding cell, the constructor of `Iterator` is private, but we define `LinkedList` as a friend of `Iterator` too.

**Iterator.h:**

```cpp
class Iterator { 
  private: 
    Iterator(Cell *cellPtr); 
    friend class LinkedList; 

  public: 
    Iterator(); 
```

The third constructor is a `copy` constructor. It takes another iterator and then copies it. We cannot just accept the iterator as a parameter. Instead, we define a reference parameter. The ampersands (&) states that the parameter is a reference to an iterator object rather than an iterator object. In this way, the memory address of the iterator is sent as a parameter instead of the object itself. We also state that the object referred to is constant, so that it cannot be altered by the constructor.

In this case, it is necessary to use a reference parameter. If we had defined a simple iterator object as a parameter it would have caused indefinite circular initialization. However, in other cases, we use this technique for efficiency reasons. It takes less time and requires less memory to pass the address of the object than to copy the object itself as a parameter:

```cpp
    Iterator(const Iterator& iterator); 

    double getValue() { return m_cellPtr->getValue(); } 
    void setValue(double value) { m_cellPtr->setValue(value); } 
```

The `hasNext` methods returns `true` if the iterator has not yet reached the end of the list, and `next` moves the iterator one step forwards, towards the end of the list, as shown in the following example:

```cpp
    bool hasNext() const { return (m_cellPtr != nullptr); } 
    void next() { m_cellPtr = m_cellPtr->getNext(); } 
```

In the same way, the `hasPrevious` method returns `true` if the iterator has not yet reached the beginning of the list, and `previous` moves the iterator one step backward, to the beginning of the list:

```cpp
    bool hasPrevious() const { return (m_cellPtr != nullptr); } 
    void previous() { m_cellPtr = m_cellPtr->getPrevious(); } 

  private: 
    Cell *m_cellPtr; 
}; 
```

**Iterator.cpp:**

```cpp
#include "Cell.h" 
#include "Iterator.h" 

Iterator::Iterator(Cell *cellPtr) 
 :m_cellPtr(cellPtr) { 
  // Empty. 
}  

Iterator::Iterator() 
 :m_cellPtr(nullptr) { 
  // Empty. 
} 

Iterator::Iterator(const Iterator& iterator) 
 :m_cellPtr(iterator.m_cellPtr) { 
  // Empty. 
} 
```

# The List class

The `LinkedList` class holds methods for finding, adding, inserting, and removing values, as well as comparing lists. Moreover, it also holds methods for reading and writing the list, and iterating through the list both forwards and backwards. The linked list is in fact a double-linked list. We can follow the links of the cells in both directions: from the beginning to the end as well as backwards from the end to the beginning.

**LinkedList.h:**

```cpp
class LinkedList { 
  public: 
    LinkedList(); 
```

The `copy` constructor and the `assign` method both copies the given list:

```cpp
    LinkedList(const LinkedList& list); 
    void assign(const LinkedList& list); 
```

The destructor deallocates all memory allocated for the cells in the linked list:

```cpp
    ~LinkedList(); 

    int size() const {return m_size;} 
    bool empty() const {return (m_size == 0);} 
```

The `find` methods search for the `value`. If it finds the `value`, it returns `true` and sets `findIterator` to the position of the `value`:

```cpp
    bool find(double value, Iterator& findIterator); 
```

The `equal` and `notEqual` methods compare this linked list to the given linked list and return `true` if they are equal or not equal, respectively, as shown in the following code snippet:

```cpp
    bool equal(const LinkedList& list) const; 
    bool notEqual(const LinkedList& list) const; 
```

What if we want to add a value or another list to an existing list? The `add` methods adds a value or another list at the end of this list, and `insert` inserts a value or a list at the position given by the iterator:

```cpp
    void add(double value); 
    void add(const LinkedList& list); 

    void insert(const Iterator& insertPosition, double value); 
    void insert(const Iterator& insertPosition, 
                const LinkedList& list); 
```

The `erase` method erases the value at the given position, and `clear` erases every value in the list, as shown in the following example:

```cpp
    void erase(const Iterator& erasePosition); 
    void clear(); 
```

The `remove` method removes the values from the first iterator to the last iterator, inclusive. The second parameter is a default parameter. It means that the method can be called with one or two parameters. In case of one parameter, the second parameter is given the value in the declaration, which in this case is the `Iterator(nullptr)` that represents the position one step beyond the end of the list. This implies that when `remove` is called with one iterator, every value from that iterator, inclusive, to the end of the list are removed.  The `nullptr` pointer is in fact a special pointer that is converted to the type it points at or is compared to. In this case, a pointer to `Cell`. Informally, we can say that a point is null when it holds the value `nullptr`:

```cpp
    void remove(const Iterator& firstPosition, 
                const Iterator& lastPosition = Iterator(nullptr)); 
```

The `first` and `last` methods return iterators located at the first and last value of the list:

```cpp
    Iterator first() const { return Iterator(m_firstCellPtr); } 
    Iterator last() const { return Iterator(m_lastCellPtr); } 
```

The `read` and `write` methods read the values of the list from an input file stream and write its values to an output file stream. A file stream is used to communicate with a file. Note that the `cin` and `cout` objects, which we have used in earlier sections, are in fact input and output stream objects:

```cpp
    void read(istream& inStream); 
    void write(ostream& outStream); 
```

Similar to the queue of the earlier section, the list holds pointers to the first and last cell in the linked list:

```cpp
  private: 
    int m_size; 
    Cell *m_firstCellPtr, *m_lastCellPtr; 
}; 
```

**LinkedList.cpp:**

```cpp
#include <IOStream> 
using namespace std; 

#include "Cell.h" 
#include "Iterator.h" 
#include "List.h" 

LinkedList::LinkedList() 
 :m_size(0), 
  m_firstCellPtr(nullptr), 
  m_lastCellPtr(nullptr) { 
  // Empty. 
} 
```

The `copy` constructor simply calls `assign` to copy the values of the `list` parameter:

```cpp
LinkedList::LinkedList(const LinkedList& list) { 
  assign(list); 
} 
```

The `assign` method copies the given list into its own linked list:

```cpp
void LinkedList::assign(const LinkedList& list) { 
  m_size = 0; 
  m_firstCellPtr = nullptr; 
  m_lastCellPtr = nullptr; 
  Cell *listCellPtr = list.m_firstCellPtr;
  add(list);
} 
```

The destructor simply calls clear to deallocate all the memory allocated by the cells of the linked list:

```cpp
LinkedList::~LinkedList() { 
  clear(); 
} 
```

The `clear` method iterates through the linked list and deallocates every cell:

```cpp
void LinkedList::clear() { 
  Cell *currCellPtr = m_firstCellPtr; 
```

For each cell in the linked list, we must first save its address in `deleteCellPtr`, move forward in the linked list, and deallocate the cell. If we would simply call `delete` on `currCellPtr`, the following call to `getNext` would not work since, in that case, we would call a method of a deallocated object:

```cpp
  while (currCellPtr != nullptr) { 
    Cell *deleteCellPtr = currCellPtr; 
    currCellPtr = currCellPtr->getNext(); 
    delete deleteCellPtr; 
  } 
```

When the list has become empty, both cell pointers are null and the size is zero:

```cpp
  m_firstCellPtr = nullptr; 
  m_lastCellPtr = nullptr; 
  m_size = 0; 
} 
```

The `find` method iterates through the linked list, sets `findIterator`, and returns `true` when it has found the value. If it does not find the value, `false` is returned and `findIterator` remains unaffected. In order for this to work, `findIterator` must be a reference to an `Iterator` object rather than an `Iterator` object itself. A pointer to an `Iterator` object would also work:

```cpp
bool LinkedList::find(double value, Iterator& findIterator) { 
  Iterator iterator = first(); 

  while (iterator.hasNext()) { 
    if (value == iterator.getValue()) { 
      findIterator = iterator; 
      return true; 
    } 

    iterator.next(); 
  } 

  return false; 
} 
```

If two lists have different sizes, they are not equal. Likewise, if they have the same size, but not the same values, they are not equal:

```cpp
bool LinkedList::equal(const LinkedList& list) const { 
  if (m_size != list.m_size) { 
    return false; 
  } 

  Iterator thisIterator = first(), listIterator = list.first(); 

  while (thisIterator.hasNext()) { 
    if (thisIterator.getValue() != listIterator.getValue()) { 
      return false; 
    } 

    thisIterator.next(); 
    listIterator.next(); 
  } 
```

However, if the list holds the same size and the same values, they are equal:

```cpp
  return true; 
} 
```

When we have to decide whether two lists are not equal, we simply call `equal`. The exclamation mark (`!`) is the logical `not` operator, as shown in the following example:

```cpp
bool LinkedList::notEqual(const LinkedList& list) const { 
  return !equal(list); 
} 
```

When adding a value to the list, we dynamically allocate a cell:

```cpp
void LinkedList::add(double value) { 
  Cell *newCellPtr = new Cell(value, m_lastCellPtr, nullptr); 
```

If the first cell pointer is null, we set it to point at the new cell since the list is empty:

```cpp
  if (m_firstCellPtr == nullptr) { 
    m_firstCellPtr = newCellPtr; 
  } 
```

However, if the first cell pointer is not null, the list is not empty, and we set the next pointer of the last cell pointer to point at the new cell:

```cpp
  else { 
    m_lastCellPtr->setNext(newCellPtr); 
  } 
```

Either way, we set the last cell pointer to point at the new cell and increase the size of the list:

```cpp
  m_lastCellPtr = newCellPtr; 
  ++m_size; 
} 
```

# Adding a list to an existing list

When adding a whole list to the list, we act the same way for each value in the list as when we added a single value in `add` previously. We dynamically allocate a new cell, if the first cell pointer is null, we assign it to point at the new cell. If it is not null, we assign the last cell pointer's next-pointer to point at the new cell. Either way, we set the last cell pointer to point at a new cell:

```cpp
void LinkedList::add(const LinkedList& list) { 
  Cell *listCellPtr = list.m_firstCellPtr; 
```

The `while` statement repeats for as long as its condition is true. In this case, for as long as we have not reached the end of the list:

```cpp
  while (listCellPtr != nullptr) { 
    double value = listCellPtr->getValue(); 
    Cell *newCellPtr = new Cell(value, m_lastCellPtr, nullptr);  
```

If `m_firstList` is null, our linked list is still empty and `newCellPtr` points to the first cell of a new linked list. In that case, we let `m_firstList` point at the new cell:

```cpp
    if (m_firstCellPtr == nullptr) { 
      m_firstCellPtr = newCellPtr; 
    } 
```

If `m_firstList` is not null, our list is not empty and `m_firstList` shall not be modified. Instead, we set the next pointer of `m_lastCellPtr` to point at the new cell:

```cpp
    else {       
      m_lastCellPtr->setNext(newCellPtr); 
    } 
```

Either way, the last cell pointer is set to the new cell pointer:

```cpp
    m_lastCellPtr = newCellPtr; 
```

Finally, the list cell pointer is set to point at its next cell pointer. Eventually, the list cell pointer will be null and the `while` statement is finished:

```cpp
    listCellPtr = listCellPtr->getNext(); 
  } 

  m_size += list.m_size; 
} 
```

When inserting a value at the position given by the iterator, we set its previous pointer to point at the cell before the position in the list (which is null if the position is the first one in the list). We then check whether the first cell pointer is null in the same way as in the preceding `add` methods:

```cpp
void LinkedList::insert(const Iterator& insertPosition, 
                        double value) { 
  Cell *insertCellPtr = insertPosition.m_cellPtr; 
  Cell *newCellPtr = 
    new Cell(value, insertCellPtr->getPrevious(), insertCellPtr);  
  insertCellPtr->setPrevious(newCellPtr); 

  if (insertCellPtr == m_firstCellPtr) { 
    m_firstCellPtr = newCellPtr; 
  } 
  else { 
    newCellPtr->getPrevious()->setNext(newCellPtr); 
  } 

  ++m_size; 
} 
```

When inserting a list, we begin by checking whether the position represents the null pointer. In that case, the position is beyond the end of our list, and we just call `add` instead:

```cpp
void LinkedList::insert(const Iterator& insertPosition, 
                        const LinkedList& list) { 
  Cell *insertCellPtr = insertPosition.m_cellPtr; 

  if (insertCellPtr == nullptr) { 
    add(list); 
  } 
  else { 
    Cell *firstInsertCellPtr = nullptr, 
         *lastInsertCellPtr = nullptr, 
         *listCellPtr = list.m_firstCellPtr; 

    while (listCellPtr != nullptr) { 
      Cell *newCellPtr = new Cell(listCellPtr->getValue(), 
                                  lastInsertCellPtr, nullptr); 

      if (firstInsertCellPtr == nullptr) { 
        firstInsertCellPtr = newCellPtr; 
      } 
      else { 
        lastInsertCellPtr->setNext(newCellPtr); 
      } 

      lastInsertCellPtr = newCellPtr; 
      listCellPtr = listCellPtr->getNext(); 
    } 
```

We check whether the list to be inserted is empty by comparing `firstInsertCellPtr` with `nullptr`. Since `firstInsertCellPtr` points at the first value of the list, the list is empty if it is null:

```cpp
    if (firstInsertCellPtr != nullptr) { 
      if (insertCellPtr->getPrevious() != nullptr) { 
        insertCellPtr->getPrevious()->setNext(firstInsertCellPtr); 
        firstInsertCellPtr-> 
          setPrevious(insertCellPtr->getPrevious()); 
      } 
      else { 
        m_firstCellPtr = firstInsertCellPtr; 
      } 
    } 

    if (lastInsertCellPtr != nullptr) { 
      lastInsertCellPtr->setNext(insertCellPtr); 
      insertCellPtr->setPrevious(lastInsertCellPtr); 
    } 

    m_size += list.m_size; 
  } 
} 
```

# Erasing a value from the list

The `erase` method simply calls `remove` with the given position as both its start and end position:

```cpp
void LinkedList::erase(const Iterator& removePosition) { 
  remove(removePosition, removePosition); 
} 
```

When erasing a value from the list, we iterate through the list and deallocate the cell for each value to be removed:

```cpp
void LinkedList::remove(const Iterator& firstPosition, 
           const Iterator& lastPosition /*= Iterator(nullptr)*/) { 
  Cell *firstCellPtr = firstPosition.m_cellPtr, 
       *lastCellPtr = lastPosition.m_cellPtr; 
  lastCellPtr = (lastCellPtr == nullptr) 
                ? m_lastCellPtr : lastCellPtr; 

  Cell *previousCellPtr = firstCellPtr->getPrevious(), 
       *nextCellPtr = lastCellPtr->getNext(); 

  Cell *currCellPtr = firstCellPtr; 
  while (currCellPtr != nextCellPtr) { 
    Cell *deleteCellPtr = currCellPtr; 
    currCellPtr = currCellPtr->getNext(); 
    delete deleteCellPtr; 
    --m_size; 
  } 
```

When we have to erase the cells, we have three cases to consider. If the last cell before the first removed cell is not null, meaning that there is a part of the list remaining before the remove position, we set its next pointer to point at the first cell after the removed position. If the last cell before the first removed cell is null, we set the first cell pointer to point at that cell:

```cpp
  if (previousCellPtr != nullptr) { 
    previousCellPtr->setNext(nextCellPtr); 
  } 
  else { 
    m_firstCellPtr = nextCellPtr; 
  } 
```

We do the same thing with the position of the list remaining after the last cell to be removed. If there is a remaining part of the list left, we set its first cell's previous pointer to the last cell of the list remaining before the removed part:

```cpp
  if (nextCellPtr != nullptr) { 
    nextCellPtr->setPrevious(previousCellPtr); 
  } 
  else { 
    m_lastCellPtr = previousCellPtr; 
  } 
} 
```

When reading a list, we first read its `size`. Then we read the values:

```cpp
void LinkedList::read(istream& inStream) { 
  int size; 
  inStream >> size; 

  int count = 0; 
  while (count < size) { 
    double value; 
    inStream >> value; 
    add(value); 
    ++count; 
  } 
} 
```

When writing a list, we write the values separated by commas and enclosed by brackets ("`[`" and "`]`"):

```cpp
void LinkedList::write(ostream& outStream) { 
  outStream << "["; 
  bool firstValue = true; 

  Iterator iterator = first(); 
  while (iterator.hasNext()) { 
    outStream << (firstValue ? "" : ",") << iterator.getValue(); 
    firstValue = false; 
    iterator.next(); 
  } 

  outStream << "]"; 
} 
```

We test the list by adding some values and iterate through them, forwards and backward.

**Main.cpp:**

```cpp
#include <IOStream> 
using namespace std; 

#include "Cell.h" 
#include "Iterator.h" 
#include "List.h" 

void main() { 
  LinkedList list; 
  list.add(1); 
  list.add(2); 
  list.add(3); 
  list.add(4); 
  list.add(5); 
  list.write(cout); 
  cout << endl; 

  { Iterator iterator = list.first(); 
    while (iterator.hasNext()) { 
      cout << iterator.getValue() << " "; 
      iterator.next(); 
    } 
    cout << endl; 
  } 

  { Iterator iterator = list.last(); 
    while (iterator.hasPrevious()) { 
      cout << iterator.getValue() << " "; 
      iterator.previous(); 
    } 
    cout << endl; 
  } 
} 
```

When executing the code, the output is displayed in a command window:

![](img/12893731-9b8b-4a4b-a928-55a543679ee5.png)

# The Set class

A set is an unordered structure without duplicates. The `Set` class is a subclass of `LinkedList`. Note that the inheritance is private, causing all public and protected members of `LinkedList` to be private in `Set`.

**Set.h:**

```cpp
class Set : private LinkedList { 
  public: 
    Set(); 
    Set(double value); 
    Set(const Set& set); 
    void assign(const Set& set); 
    ~Set(); 
```

The `equal` method returns `true` if the set has the values. Note that we do not care about any order in the set:

```cpp
    bool equal(const Set& set) const; 
    bool notEqual(const Set& set) const; 
```

The `exists` method returns `true` if the given value, or each value in the given set, respectively, is present:

```cpp
    bool exists(double value) const; 
    bool exists(const Set& set) const; 
```

The `insert` method inserts the given value or each value of the given set. It only inserts values not already present in the set, since a set holds no duplicates:

```cpp
    bool insert(double value); 
    bool insert(const Set& set); 
```

The `remove` method removes the given value or each value of the given set, if present:

```cpp
    bool remove(double value); 
    bool remove(const Set& set); 
```

The `size`, `empty`, and `first` methods simply call their counterparts in `LinkedList`. Since there is no order in a set it would be meaningless to also override `end` in `LinkedList`:

```cpp
    int size() const { return LinkedList::size(); } 
    bool empty() const { return LinkedList::empty(); } 
    Iterator first() const { return LinkedList::first(); } 
```

The `unionSet`, `intersection`, and `difference` free-standing functions are friends to `Set`, which means that they have access to all private and protected members of `Set`.

We cannot name the `unionSet` method `union` since it is a keyword in C++.

Note that when a method in a class is marked as a `friend`, it is in fact not a method of that class, but rather a function:

```cpp
    friend Set unionSet(const Set& leftSet, const Set& rightSet); 
    friend Set intersection(const Set& leftSet, 
                            const Set& rightSet); 
    friend Set difference(const Set& leftSet, 
                          const Set& rightSet); 
```

The `read` and `write` methods read and write the set in the same way as their counterparts in `LinkedList`:

```cpp
    void read(istream& inStream); 
    void write(ostream& outStream); 
}; 
```

The `unionSet`, `intersection`, and `difference` functions that were friends of `Set` are declared outside the class definition:

```cpp
Set unionSet(const Set& leftSet, const Set& rightSet);
Set intersection(const Set& leftSet, const Set& rightSet);
Set difference(const Set& leftSet, const Set& rightSet);
```

**Set.cpp:**

```cpp
#include <IOStream> 
using namespace std; 

#include "..\ListBasic\Cell.h" 
#include "..\ListBasic\Iterator.h" 
#include "..\ListBasic\List.h" 
#include "Set.h" 
```

The constructors call their counterparts in `LinkedList`. The default constructor (without parameters) calls, in fact, the default constructor of `LinkedList` implicitly:

```cpp
Set::Set() { 
  // Empty. 
} 

Set::Set(double value) { 
  add(value); 
} 

Set::Set(const Set& set) 
 :LinkedList(set) { 
  // Empty. 
} 
```

The destructor calls implicitly its counterparts in `LinkedList`, which deallocates the memory associated with the values of the set. In this case, we could have omitted the destructor, and the destructor of `LinkedList` would still be called using the following code:

```cpp
Set::~Set() { 
  // Empty. 
} 
```

The `assign` method simply clears the set and adds the given set:

```cpp
void Set::assign(const Set& set) { 
  clear(); 
  add(set); 
} 
```

The sets are equal if they have the same `size`, and if every value of one set is present in the other set. In that case, every value of the other set must also be present in the first set:

```cpp
bool Set::equal(const Set& set) const { 
  if (size() != set.size()) { 
    return false; 
  } 

  Iterator iterator = first(); 
  while (iterator.hasNext()) { 
    if (!set.exists(iterator.getValue())) { 
      return false; 
    } 

    iterator.next(); 
  } 

  return true;          
} 

bool Set::notEqual(const Set& set) const { 
  return !equal(set); 
} 
```

The `exists` method uses the iterator of `LinkedList` to iterate through the set. It returns `true` if it finds the value:

```cpp
bool Set::exists(double value) const { 
  Iterator iterator = first(); 

  while (iterator.hasNext()) { 
    if (value == iterator.getValue()) { 
      return true; 
    } 

    iterator.next(); 
  } 

  return false; 
} 
```

The second `exists` method iterates through the given set and returns `false` if any of its values are not present in the set. It returns `true` if all its values are present in the set:

```cpp
bool Set::exists(const Set& set) const { 
  Iterator iterator = set.first(); 

  while (iterator.hasNext()) { 
    if (!exists(iterator.getValue())) { 
      return false; 
    } 

    iterator.next(); 
  } 

  return true; 
} 
```

The first `insert` method adds the value if it is not already present in the set:

```cpp
bool Set::insert(double value) { 
  if (!exists(value)) { 
    add(value); 
    return true; 
  } 

  return false; 
} 
```

The second `insert` method iterates through the given set and inserts every value by calling the first insert method. In this way, each value not already present in the set is inserted:

```cpp
bool Set::insert(const Set& set) { 
  bool inserted = false; 
  Iterator iterator = set.first(); 

  while (iterator.hasNext()) { 
    double value = iterator.getValue(); 

    if (insert(value)) { 
      inserted = true; 
    } 

    iterator.next(); 
  } 

  return inserted; 
} 
```

The first `remove` method removes the value and returns `true` if it is present in the set. If it is not present, it returns `false`:

```cpp
bool Set::remove(double value) { 
  Iterator iterator; 

  if (find(value, iterator)) { 
    erase(iterator); 
    return true; 
  } 

  return false; 
} 
```

The second `remove` method iterates through the given set and removes each of its values. It returns `true` if at least one value is removed:

```cpp
bool Set::remove(const Set& set) { 
  bool removed = false; 
  Iterator iterator = set.first(); 

  while (iterator.hasNext()) { 
    double value = iterator.getValue(); 

    if (remove(value)) { 
      removed = true; 
    } 

    iterator.next(); 
  } 

  return removed; 
} 
```

# Union, intersection, and difference operations

The `unionSet` function creates a resulting set initialized with the left-hand set and then adds the right-hand set:

```cpp
Set unionSet(const Set& leftSet, const Set& rightSet) { 
  Set result(leftSet); 
  result.insert(rightSet); 
  return result; 
} 
```

The `intersection` method is a little bit more complicated than the `union` or `difference` methods. The intersection of two sets, A and B, can be defined as the difference between their union and their differences:

*A∩B=(A∪B)-((A-B)-(B-A))*

```cpp
Set intersection(const Set& leftSet, const Set& rightSet) { 
  return difference(difference(unionSet(leftSet, rightSet), 
                               difference(leftSet, rightSet)), 
                    difference(rightSet, leftSet)); 
} 
```

The `difference` method creates a result set with the left-hand set and then removes the right-hand set:

```cpp
Set difference(const Set& leftSet, const Set& rightSet) { 
  Set result(leftSet); 
  result.remove(rightSet); 
  return result; 
} 
```

The `read` method is similar to its counterpart in `LinkedList`. However, `insert` is called instead of `add`. In this way, no duplicates are inserted in the set:

```cpp
void Set::read(istream& inStream) { 
  int size; 
  inStream >> size; 

  int count = 0; 
  while (count < size) { 
    double value; 
    inStream >> value; 
    insert(value); 
    ++count; 
  } 
} 
```

The `write` method is also similar to its counterpart in `LinkedList`. However, the set is enclosed in brackets ("`{`" and "`}`") instead of squares ("`[`" and "`]`"):

```cpp
void Set::write(ostream& outStream) { 
  outStream << "{"; 
  bool firstValue = true; 
  Iterator iterator = first(); 

  while (iterator.hasNext()) { 
    outStream << (firstValue ? "" : ",") << iterator.getValue(); 
    firstValue = false; 
    iterator.next(); 
  } 

  outStream << "}"; 
} 
```

We test the set by letting the user input two sets and evaluate their union, intersection, and difference.

**Main.cpp:**

```cpp
#include <IOStream> 
using namespace std; 

#include "..\ListBasic\Cell.h" 
#include "..\ListBasic\Iterator.h" 
#include "..\ListBasic\List.h" 
#include "Set.h" 

void main() { 
  Set s, t; 
  s.read(cin); 
  t.read(cin); 

  cout << endl << "s = "; 
  s.write(cout); 
  cout << endl; 

  cout << endl << "t = "; 
  t.write(cout); 
  cout << endl << endl; 

  cout << "union: "; 
  unionSet(s, t).write(cout); 
  cout << endl; 

  cout << "intersection: "; 
  unionSet(s, t).write(cout); 
  cout << endl; 

  cout << "difference: "; 
  unionSet(s, t).write(cout); 
  cout << endl; 
} 
```

# Basic searching and sorting

In this chapter, we will also study some searching and sorting algorithms. When searching for a value with linear search we simply go through the list from its beginning to its end. We return the zero-based index of the value, or minus one if it was not found.

**Search.h:**

```cpp
int linarySearch(double value, const LinkedList& list); 
```

**Search.cpp:**

```cpp
#include <IOStream> 
using namespace std; 

#include "..\ListBasic\Cell.h" 
#include "..\ListBasic\Iterator.h" 
#include "..\ListBasic\List.h" 
#include "Search.h" 

int linarySearch(double value, const LinkedList& list) { 
  int index = 0; 
```

We use the `first` method of the list to obtain the iterator that we use to go through the list; `hasNext` returns `true` as long as there is another value in the list and `next` moves the iterator one step forward in the list:

```cpp
  Iterator iterator = list.first(); 

  while (iterator.hasNext()) { 
    if (iterator.getValue() == value) { 
      return index; 
    } 

    ++index; 
    iterator.next(); 
  } 

  return -1; 
} 
```

Now we study the select sort, insert sort, and bubble sort algorithms. Note that they take a reference to the list, not the list itself, a parameter in order for the list to become changed. Also note that the reference is not constant in these cases; if it was constant we would not be able to sort the list.

**Sort.h:**

```cpp
void selectSort(LinkedList& list); 
void insertSort(LinkedList& list); 
void bubbleSort(LinkedList& list); 
```

**Sort.cpp:**

```cpp
#include <IOStream> 
using namespace std; 

#include "..\ListBasic\Cell.h" 
#include "..\ListBasic\Iterator.h" 
#include "..\ListBasic\List.h" 
#include "Sort.h" 

void insert(double value, LinkedList& list); 
void swap(Iterator iterator1, Iterator iterator2); 
```

# The select sort algorithm

The select sort algorithm is quite simple, we iterate through the list repeatedly until it becomes empty. For each iteration, we found the smallest value, which we remove from the list and add to the resulting list. In this way, the resulting list will eventually be filled with the same values as the list. As the values were selected in order, the resulting list is sorted. Finally, we assign the resulting list to the original list:

```cpp
void selectSort(LinkedList& list) { 
  LinkedList result; 

  while (!list.empty()) { 
    Iterator minIterator = list.first(); 
    double minValue = minIterator.getValue(); 

    Iterator iterator = list.first(); 

    while (iterator.hasNext()) { 
      if (iterator.getValue() < minValue) { 
        minIterator = iterator; 
        minValue = iterator.getValue(); 
      } 

      iterator.next(); 
    } 

    list.erase(minIterator); 
    result.add(minValue); 
  } 

  list.assign(result); 
} 
```

# The insert sort algorithm

In the insert sort algorithm, we iterate through the list, and for each value we insert it at its appropriate location in the resulting list. Then we assign the resulting list to the original list:

```cpp
void insertSort(LinkedList& list) { 
  LinkedList result; 
  Iterator iterator = list.first(); 

  while (iterator.hasNext()) { 
    insert(iterator.getValue(), result); 
    iterator.next(); 
  } 

  list.assign(result); 
} 
```

The `insert` function takes a list and a value and places the value at its correct location in the list. It iterates through the list and places the value before the first value that it is less. If there is no such value in the list, the value is added at the end of the list:

```cpp
void insert(double value, LinkedList& list) { 
  Iterator iterator = list.first(); 

  while (iterator.hasNext()) { 
    if (value < iterator.getValue()) { 
      list.insert(iterator, value); 
      return; 
    } 

    iterator.next(); 
  } 

  list.add(value); 
} 
```

# The bubble sort algorithm

The bubble sort algorithm compares the values pairwise and lets them change place if they occur in the wrong order. After the first iteration, we know that the largest value is located at the end of the list. Therefore, we do not need to iterate through the whole list the second time, we can omit the last value. In this way, we iterate through the list at most the number of the values in the list minus one, because when all values except the first one is at it's correct location, the first one is also at its correct location. However, the list may be properly sorted before that. Therefore, we check after each iteration if any pair of values has been swapped. If they have not, the list has been properly sorted and we exit the algorithm:

```cpp
void bubbleSort(LinkedList& list) { 
  int listSize = list.size(); 

  if (listSize > 1) { 
    int currSize = listSize - 1; 
    int outerCount = 0; 
    while (outerCount < (listSize - 1)) { 
      Iterator currIterator = list.first(); 
      Iterator nextIterator = currIterator; 
      nextIterator.next(); 
      bool changed = false; 

      int innerCount = 0; 
      while (innerCount < currSize) { 
        if (currIterator.getValue() > nextIterator.getValue()) { 
          swap(currIterator, nextIterator); 
          changed = true; 
        } 

        ++innerCount; 
        currIterator.next(); 
        nextIterator.next(); 
      } 

      if (!changed) { 
        break; 
      } 

      --currSize; 
      ++outerCount; 
    } 
  } 
} 
```

The `swap` function swaps the values at the locations given by the iterators:

```cpp
void swap(Iterator iterator1, Iterator iterator2) { 
  double tempValue = iterator1.getValue(); 
  iterator1.setValue(iterator2.getValue()); 
  iterator2.setValue(tempValue); 
} 
```

We test the algorithms by adding some values to a list, and then sort the list.

**Main.cpp:**

```cpp
#include <IOStream> 
#include <CStdLib> 

using namespace std; 

#include "..\ListBasic\Cell.h" 
#include "..\ListBasic\Iterator.h" 
#include "..\ListBasic\List.h" 

#include "Search.h" 
#include "Sort.h" 

void main() { 
  cout << "LinkedList" << endl; 

  LinkedList list; 
  list.add(9); 
  list.add(7); 
  list.add(5); 
  list.add(3); 
  list.add(1); 

  list.write(cout); 
  cout << endl; 
```

We use the `iterator` class to go through the list and call `linarySearch` for each value in the list:

```cpp
  Iterator iterator = list.first(); 
  while (iterator.hasNext()) { 
    cout << "<" << iterator.getValue() << "," 
         << linarySearch(iterator.getValue(), list) << "> "; 
    iterator.next(); 
  } 
```

We also test the search algorithm for values not present in the list, their indexes will be minus one:

```cpp
  cout << "<0," << linarySearch(0, list) << "> "; 
  cout << "<6," << linarySearch(6, list) << "> "; 
  cout << "<10," << linarySearch(10, list) << ">" 
       << endl; 
```

We sort the list by the bubble sort, select sort, and insert sort algorithms:

```cpp
  cout << "Bubble Sort "; 
  bubbleSort(list); 
  list.write(cout); 
  cout << endl; 

  cout << "Select Sort "; 
  selectSort(list); 
  list.write(cout); 
  cout << endl; 

  cout << "Insert Sort "; 
  insertSort(list); 
  list.write(cout); 
  cout << endl; 
} 
```

One way to classify searching and sorting algorithms is to use the big O notation. Informally speaking, the notation focuses on the worst-case scenario. In the insert sort case, we iterate through the list once for each value, and for each value, we may have to iterate through the whole list to find its correct location. Likewise, in the select sort case we iterate through the list once for each value, and for each value, we may need to iterate through the whole list.

Finally, in the bubble sort case, we iterate through the list once for each value and we may have to iterate through the whole list for each value. In all three cases, we may have to perform *n*² operations on a list of *n* values. Therefore, the insert, select, and bubble sort algorithms have the big-O *n*², or O (*n*²) with regards to their time efficiency. However, when it comes to their space efficiency, bubble sort is better since it operates on the same list, while insert and select sort demand an extra list for the resulting sorted list.

# The extended List class

In this section, we will revisit the `LinkedList` class. However, we will expand it in several ways:

*   The `Cell` class had a set of `set` and `get` methods. Instead, we will replace each pair with a pair of overloaded reference methods.
*   The previous list could only store values of the type `double`. Now we will define the list to be `template`, which allows it to store values of arbitrary types.
*   We will replace some of the methods with overloaded operators*.*
*   `Cell` and `Iterator` were free-standing classes. Now we will let them be inner classes, defined inside `LinkedList`.

**List.h:**

```cpp
class OutOfMemoryException : public exception { 
  // Empty. 
}; 
```

In the classes of the earlier sections, the list stored values of the type `double`. However, in these classes, instead of `double` we use the template type `T`, which is a generic type that can be instantiated by any arbitrary type. The `LinkedList` class of this section is `template`, with the generic type `T`:

```cpp
template <class T> 
class LinkedList { 
  private: 
    class Cell { 
      private: 
        Cell(const T& value, Cell* previous, Cell* next); 
```

The `value` method is overloaded in two versions. The first version is constant and returns a constant value. The other version is not constant and returns a reference to the value. In this way, it is possible to assign values to the cell's value, as shown in the following example:

```cpp
      public: 
        const T value() const { return m_value; } 
        T& value() { return m_value; } 
```

The `Cell*&` construct means that the methods return a reference to a pointer to a `Cell` object. That reference can then be used to assign a new value to the pointer:

```cpp
        const Cell* previous() const { return m_previous; } 
        Cell*& previous() { return m_previous; } 

        const Cell* next() const { return m_next; } 
        Cell*& next() { return m_next; } 

        friend class LinkedList; 

    private: 
      T m_value; 
      Cell *m_previous, *m_next; 
  }; 

  public: 
    class Iterator { 
      public: 
        Iterator(); 

      private: 
        Iterator(Cell* cellPtr); 

      public: 
        Iterator(const Iterator& iterator); 
        Iterator& operator=(const Iterator& iterator); 
```

Instead of `equal` and `notEqual`, we overload the equal and not-equal operators:

```cpp
        bool operator==(const Iterator& iterator); 
        bool operator!=(const Iterator& iterator); 
```

We also replace the increment and decrement methods with the increment (`++`) and decrement (`--`) operators. They come in two versions each—prefix and postfix. The version without parameters is the prefix version (`++i` and `--i`) and the version with an integer parameter is the postfix version (`i++` and `i--`). Note that we actually do not pass an integer parameter to the operator. The parameter is included only to distinguish between the two versions, and is ignored by the compiler:

```cpp
        bool operator++();    // prefix: ++i 
        bool operator++(int); // postfix: i++ 

        bool operator--();    // prefix: --i 
        bool operator--(int); // postfix: i-- 
```

We replace the `getValue` and `setValue` methods with two overloaded dereference operators (`*`). They work in a way similar to the `value` methods in the preceding `Cell` class. The first version is constant and returns a value, while the second version is not constant and returns a reference to the value:

```cpp
        T operator*() const; 
        T& operator*(); 

        friend class LinkedList; 

      private: 
        Cell *m_cellPtr; 
    }; 
```

# The ReverseIterator class

In order to iterate from the end to the beginning, as well as from the beginning to the end, we add `ReverseIterator`. It is nearly identical to `Iterator` used previously; the only difference is that the increment and decrement operators move in opposite directions:

```cpp
    class ReverseIterator { 
      public: 
        ReverseIterator(); 

      private: 
        ReverseIterator(Cell* cellPtr); 

      public: 
        ReverseIterator(const ReverseIterator& iterator); 
        const ReverseIterator& 
              operator=(const ReverseIterator& iterator); 

        bool operator==(const ReverseIterator& iterator); 
        bool operator!=(const ReverseIterator& iterator); 

        bool operator++();    // prefix: ++i 
        bool operator++(int); // postfix: i++ 

        bool operator--(); 
        bool operator--(int); 

        T operator*() const; 
        T& operator*(); 

        friend class LinkedList; 

      private: 
        Cell *m_cellPtr; 
  }; 

  public: 
    LinkedList(); 
    LinkedList(const LinkedList& list); 
    LinkedList& operator=(const LinkedList& list); 
    ~LinkedList(); 
    void clear(); 

    int size() const {return m_size;} 
    bool empty() const {return (m_size == 0);} 

    bool operator==(const LinkedList& list) const; 
    bool operator!=(const LinkedList& list) const; 

    void add(const T& value); 
    void add(const LinkedList& list); 

    void insert(const Iterator& insertPosition, const T& value); 
    void insert(const Iterator& insertPosition, 
                const LinkedList& list); 

    void erase(const Iterator& erasePosition); 
    void remove(const Iterator& firstPosition, 
                const Iterator& lastPosition = Iterator(nullptr)); 
```

In the earlier section, there was only the `first` and `last` methods, which return an iterator. In this section, the `begin` and `end` methods are used for forward iteration, while `rbegin` and `rend` (stands for reverse begin and reverse end) are used for backward iteration:

```cpp
    Iterator begin() const { return Iterator(m_firstCellPtr); } 
    Iterator end() const { return Iterator(nullptr); } 
    ReverseIterator rbegin() const 
      {return ReverseIterator(m_lastCellPtr);} 
    ReverseIterator rend() const 
      { return ReverseIterator(nullptr); } 
```

We replace the `read` and `write` methods with overloaded input and output stream operators. Since they are functions rather than methods, they need their own template markings:

```cpp
    template <class U> 
    friend istream& operator>>(istream& outStream, 
                               LinkedList<U>& list); 

    template <class U> 
    friend ostream& operator<<(ostream& outStream, 
                               const LinkedList<U>& list); 

  private: 
    int m_size; 
    Cell *m_firstCellPtr, *m_lastCellPtr; 
}; 
```

Note that when we implement the methods of a `template` class, we do so in the header file. Consequently, we do not need an implementation file when implementing a `template` class.

Similar to the class definitions, the method definitions must be preceded by the `template` keyword. Note that the class name `LinkedList` is followed by the type marker `<T>`:

```cpp
template <class T> 
LinkedList<T>::Cell::Cell(const T& value, Cell* previous, 
                          Cell* next) 
 :m_value(value), 
  m_previous(previous), 
  m_next(next) { 
  // Empty. 
} 

template <class T> 
LinkedList<T>::Iterator::Iterator() 
 :m_cellPtr(nullptr) { 
  // Empty. 
}  
```

Note that when we implement a method of an inner class, we need to include both the names of the outer class (`LinkedList`) and inner class (`Cell`) in the implementation:

```cpp
template <class T> 
LinkedList<T>::Iterator::Iterator(Cell* cellPtr) 
 :m_cellPtr(cellPtr) { 
  // Empty. 
}  

template <class T> 
LinkedList<T>::Iterator::Iterator(const Iterator& position) 
 :m_cellPtr(position.m_cellPtr) { 
  // Empty. 
} 
```

Since `LinkedList` is a `template` class, it is not known to the compiler that its inner class `Iterator` is, in fact, a class. As far as the compiler knows, the iterator could be a type, a value, or a class. Therefore, we need to inform the compiler by using the `typename` keyword:

```cpp
template <class T> 
typename LinkedList<T>::Iterator& 
LinkedList<T>::Iterator::operator=(const Iterator& iterator) { 
  m_cellPtr = iterator.m_cellPtr; 
  return *this; 
} 
```

The following operator versions are implemented in the same way as its method counterparts in the previous version of `LinkedList`. That is, the `equal` method has been replaced by the equation operator (`operator==`), and the `notEqual` method has been replaced by the not-equal operator (`operator!=`):

```cpp
template <class T> 
bool LinkedList<T>::Iterator::operator==(const Iterator&position){ 
  return (m_cellPtr == position.m_cellPtr); 
} 

template <class T> 
bool LinkedList<T>::Iterator::operator!=(const Iterator&position){ 
  return !(*this == position); 
} 
```

The increase operator has been replaced with both the prefix and postfix version of `operator++`. The difference between them is that the prefix version does not take any parameters, while the postfix version takes a single integer value as parameter. Note that the integer value is not used by the operator. Its value is undefined (however, it is usually set to zero) and is always ignored. It is present only to distinguish between the prefix and postfix cases:

```cpp
template <class T> 
bool LinkedList<T>::Iterator::operator++() { 
  if (m_cellPtr != nullptr) { 
    m_cellPtr = m_cellPtr->next(); 
    return true; 
  } 

  return false; 
} 

template <class T> 
bool LinkedList<T>::Iterator::operator++(int) { 
  if (m_cellPtr != nullptr) { 
    m_cellPtr = m_cellPtr->next(); 
    return true; 
  } 

  return false; 
} 
```

The `decrease` operator also comes in a prefix and a postfix version, and works in a way similar to the `increase` operator:

```cpp
template <class T> 
bool LinkedList<T>::Iterator::operator--() { 
  if (m_cellPtr != nullptr) { 
    m_cellPtr = m_cellPtr->previous(); 
    return true; 
  } 

  return false; 
} 

template <class T> 
bool LinkedList<T>::Iterator::operator--(int) { 
  if (m_cellPtr != nullptr) { 
    m_cellPtr = m_cellPtr->previous(); 
    return true; 
  } 

  return false; 
} 
```

The dereference operator also comes in two versions. The first version is constant and returns a value. The second version is not constant and returns a reference to the value, instead of the value itself. In this way, the first version can be called on a constant object, in which case we are not allowed to change its value. The second version can be called on a non-constant object only, we can change the value by assigning a new value to the value returned by the method:

```cpp
template <class T> 
T LinkedList<T>::Iterator::operator*() const { 
  return m_cellPtr->value(); 
} 

template <class T> 
T& LinkedList<T>::Iterator::operator*() { 
  return m_cellPtr->value(); 
} 
```

There are three constructors of the `ReverseIterator` class. The first constructor is a default constructor, the second constructor is initialized with a `Cell` pointer, and the third constructor is a `copy` constructor. It takes a reference to another `ReverseIterator` object, and initializes the `Cell` pointer:

```cpp
template <class T> 
LinkedList<T>::ReverseIterator::ReverseIterator() 
 :m_cellPtr(nullptr) { 
  // Empty. 
}  

template <class T> 
LinkedList<T>::ReverseIterator::ReverseIterator(Cell* currCellPtr) 
 :m_cellPtr(currCellPtr) { 
  // Empty. 
}  

template <class T> 
LinkedList<T>::ReverseIterator::ReverseIterator 
                                (const ReverseIterator& position) 
 :m_cellPtr(position.m_cellPtr) { 
  // Empty. 
} 
```

The equality operator initializes the `Cell` pointer with the `Cell` pointer of the given `ReverseIterator` object reference:

```cpp
template <class T> 
const typename LinkedList<T>::ReverseIterator& 
LinkedList<T>::ReverseIterator::operator=(const ReverseIterator& position) { 
  m_cellPtr = position.m_cellPtr; 
  return *this; 
} 
```

Two reverse iterators are equal if their cell pointers point at the same cell:

```cpp
template <class T> 
bool LinkedList<T>::ReverseIterator::operator== 
                           (const ReverseIterator& position) { 
  return (m_cellPtr == position.m_cellPtr); 
} 

template <class T> 
bool LinkedList<T>::ReverseIterator::operator!= 
                           (const ReverseIterator& position) { 
  return !(*this == position); 
} 
```

The difference between the increase and decrease operators of the `Iterator` and `ReverseIterator` classes is that in `Iterator` the increment operators calls next and the `decrement` operators call `previous` in `Cell`. In `ReverseIterator` it is the other way around: the increment operators call `previous` and the decrement operators call `next`. As the names implies: `Iterator` iterates forward, while `ReverseIterator` iterates backwards:

```cpp
template <class T> 
bool LinkedList<T>::ReverseIterator::operator++() { 
  if (m_cellPtr != nullptr) { 
    m_cellPtr = m_cellPtr->previous(); 
    return true; 
  } 

  return false; 
} 

template <class T> 
bool LinkedList<T>::ReverseIterator::operator++(int) { 
  if (m_cellPtr != nullptr) { 
    m_cellPtr = m_cellPtr->previous(); 
    return true; 
  } 

  return false; 
} 

template <class T> 
bool LinkedList<T>::ReverseIterator::operator--() { 
  if (m_cellPtr != nullptr) { 
    m_cellPtr = m_cellPtr->next(); 
    return true; 
  } 

  return false; 
} 

template <class T> 
bool LinkedList<T>::ReverseIterator::operator--(int) { 
  if (m_cellPtr != nullptr) { 
    m_cellPtr = m_cellPtr->next(); 
    return true; 
  } 

  return false; 
} 

template <class T> 
T LinkedList<T>::ReverseIterator::operator*() const { 
  return m_cellPtr->value(); 
} 

template <class T> 
T& LinkedList<T>::ReverseIterator::operator*() { 
  return m_cellPtr->value(); 
} 
```

The default constructor of `LinkedList` initializes the list to become empty, with the pointer to the first and last cell set to null:

```cpp
template <class T> 
LinkedList<T>::LinkedList() 
 :m_size(0), 
  m_firstCellPtr(nullptr), 
  m_lastCellPtr(nullptr) { 
  // Empty. 
} 

template <class T> 
LinkedList<T>::LinkedList(const LinkedList<T>& list) { 
 *this = list; 
} 
```

The assignment operator copies the values of the given list, in the same way as the non-template method:

```cpp
template <class T> 
LinkedList<T>& LinkedList<T>::operator=(const LinkedList<T>&list){ 
  m_size = 0; 
  m_firstCellPtr = nullptr; 
  m_lastCellPtr = nullptr; 

  if (list.m_size > 0) { 
    for (Cell *listCellPtr = list.m_firstCellPtr, 
              *nextCellPtr = list.m_lastCellPtr->next(); 
         listCellPtr != nextCellPtr; 
         listCellPtr = listCellPtr->next()) { 
      Cell *newCellPtr = new Cell(listCellPtr->value(), 
                                  m_lastCellPtr, nullptr); 

      if (m_firstCellPtr == nullptr) { 
        m_firstCellPtr = newCellPtr; 
      } 
```

Note that we use the reference version of the `next` method, which allows us to assign values to the method call. Since `next` returns a reference to the next pointer of the cell, we can assign value of `newCellPtr` to that pointer:

```cpp
      else { 
        m_lastCellPtr->next() = newCellPtr; 
      } 

      m_lastCellPtr = newCellPtr; 
      ++m_size; 
    } 
  } 

  return *this; 
} 
```

The destructor simply calls the `clear` method, which goes through the linked list and deletes every cell:

```cpp
template <class T> 
LinkedList<T>::~LinkedList() { 
  clear(); 
} 

template <class T> 
void LinkedList<T>::clear() { 
  Cell *currCellPtr = m_firstCellPtr; 

  while (currCellPtr != nullptr) { 
    Cell *deleteCellPtr = currCellPtr; 
    currCellPtr = currCellPtr->next(); 
    delete deleteCellPtr; 
  } 
```

When the cells are deleted, the pointer to the first and last cell is set to null:

```cpp
  m_size = 0; 
  m_firstCellPtr = nullptr; 
  m_lastCellPtr = nullptr; 
} 
```

Two lists are equal if they have the same size, and if their cells hold the same values:

```cpp
template <class T> 
bool LinkedList<T>::operator==(const LinkedList<T>& list) const { 
  if (m_size != list.m_size) { 
    return false; 
  } 

  for (Iterator thisIterator = begin(), 
                listIterator = list.begin(); 
       thisIterator != end(); ++thisIterator, ++listIterator) { 
    if (*thisIterator != *listIterator) { 
      return false; 
    } 
  } 

  return true; 
} 

template <class T> 
bool LinkedList<T>::operator!=(const LinkedList<T>& list) const { 
  return !(*this == list); 
} 
```

The `add` method adds a cell with a new value at the end of the list, as shown in the following example:

```cpp
template <class T> 
void LinkedList<T>::add(const T& value) { 
  Cell *newCellPtr = new Cell(value, m_lastCellPtr, nullptr); 

  if (m_lastCellPtr == nullptr) { 
    m_firstCellPtr = newCellPtr; 
    m_lastCellPtr = newCellPtr; 
  } 
  else { 
    m_lastCellPtr->next() = newCellPtr; 
    m_lastCellPtr = newCellPtr; 
  } 

  ++m_size; 
} 
```

The second version of `add` adds the given list at the end of the list, as shown in the following example:

```cpp
template <class T> 
void LinkedList<T>::add(const LinkedList<T>& list) { 
  for (Cell *listCellPtr = list.m_firstCellPtr; 
       listCellPtr != nullptr; listCellPtr = listCellPtr->next()){ 
    const T& value = listCellPtr->value(); 
    Cell *newCellPtr = new Cell(value, m_lastCellPtr, nullptr); 

    if (m_lastCellPtr == nullptr) { 
      m_firstCellPtr = newCellPtr; 
    } 
    else {       
      m_lastCellPtr->next() = newCellPtr; 
    } 

    m_lastCellPtr = newCellPtr; 
  } 

  m_size += list.m_size; 
} 
```

The `insert` method adds a value or a list at the given position:

```cpp
template <class T> 
void LinkedList<T>::insert(const Iterator& insertPosition, 
                           const T& value) { 
  if (insertPosition.m_cellPtr == nullptr) { 
    add(value); 
  } 
  else { 
    Cell *insertCellPtr = insertPosition.m_cellPtr; 
    Cell *newCellPtr = 
      new Cell(value, insertCellPtr->previous(), insertCellPtr); 

    insertCellPtr->previous() = newCellPtr; 

    if (insertCellPtr == m_firstCellPtr) { 
      m_firstCellPtr = newCellPtr; 
    } 
    else { 
      newCellPtr->previous()->next() = newCellPtr; 
    } 

    ++m_size; 
  } 
} 

template <class T> 
void LinkedList<T>::insert(const Iterator& insertPosition, 
                           const LinkedList<T>& list) { 
  if (insertPosition.m_cellPtr == nullptr) { 
    add(list); 
  } 
  else { 
    Cell *insertCellPtr = insertPosition.m_cellPtr; 

    Cell *firstInsertCellPtr = nullptr, 
         lastInsertCellPtr = nullptr; 
    for (Cell *listCellPtr = list.m_firstCellPtr; 
         listCellPtr != nullptr;listCellPtr=listCellPtr->next()) { 
      double value = listCellPtr->value(); 
      Cell *newCellPtr = 
        new Cell(value, lastInsertCellPtr, nullptr); 

      if (firstInsertCellPtr == nullptr) { 
        firstInsertCellPtr = newCellPtr; 
      } 
      else { 
        lastInsertCellPtr->next() = newCellPtr; 
      } 

      lastInsertCellPtr = newCellPtr; 
    } 

    if (firstInsertCellPtr != nullptr) { 
      if (insertCellPtr->previous() != nullptr) { 
        insertCellPtr->previous()->next() = firstInsertCellPtr; 
        firstInsertCellPtr->previous() = 
          insertCellPtr->previous(); 
      } 
      else { 
        m_firstCellPtr = firstInsertCellPtr; 
      } 
    } 

    if (lastInsertCellPtr != nullptr) { 
      lastInsertCellPtr->next() = insertCellPtr; 
      insertCellPtr->previous() = lastInsertCellPtr; 
    } 

    m_size += list.m_size; 
  } 
} 

```

The `erase` and `remove` methods remove a value of a sub-list from the list:

```cpp
template <class T> 
void LinkedList<T>::erase(const Iterator& removePosition) { 
  remove(removePosition, removePosition); 
} 

template <class T> 
void LinkedList<T>::remove(const Iterator& firstPosition, 
           const Iterator& lastPosition /*= Iterator(nullptr)*/) { 
  Cell *firstCellPtr = firstPosition.m_cellPtr, 
      *lastCellPtr = lastPosition.m_cellPtr; 
  lastCellPtr = (lastCellPtr == nullptr) 
                ? m_lastCellPtr : lastCellPtr; 

  Cell *previousCellPtr = firstCellPtr->previous(), 
      *nextCellPtr = lastCellPtr->next(); 

  Cell *currCellPtr = firstCellPtr; 
  while (currCellPtr != nextCellPtr) { 
    Cell *deleteCellPtr = currCellPtr; 
    currCellPtr = currCellPtr->next(); 
    delete deleteCellPtr; 
    --m_size; 
  } 

  if (previousCellPtr != nullptr) { 
    previousCellPtr->next() = nextCellPtr;   
  } 
  else { 
    m_firstCellPtr = nextCellPtr; 
  } 

  if (nextCellPtr != nullptr) { 
    nextCellPtr->previous() = previousCellPtr; 
  } 
  else { 
    m_lastCellPtr = previousCellPtr; 
  } 
} 
```

The input stream operator first reads the `size` of the list, and then the values themselves:

```cpp
template <class T> 
istream& operator>>(istream& inStream, LinkedList<T>& list) { 
  int size; 
  inStream >> size; 

  for (int count = 0; count < size; ++count) { 
    T value; 
    inStream >> value; 
    list.add(value); 
  } 

  return inStream; 
} 
```

The output stream operator writes the list on the given stream, surrounded by brackets and with the values separated by commas:

```cpp
template <class T> 
ostream& operator<<(ostream& outStream,const LinkedList<T>& list){ 
  outStream << "["; 

  bool first = true; 
  for (const T& value : list) { 
    outStream << (first ? "" : ",") << value; 
    first = false; 
  } 

  outStream << "]"; 
  return outStream; 
} 
```

We test the `LinkedList` class by letting the user input a list that we iterate automatically with the `for` statement, as well as manually with forward and backward iterators.

**Main.cpp:**

```cpp
#include <IOStream> 
#include <Exception> 
using namespace std; 

#include "List.h" 

void main() { 
  LinkedList<double> list; 
  cin >> list; 
  cout << list <&amp;lt; endl; 
```

Note that it is possible to use the `for` statement directly on the list since the extended list holds the `begin` method, which returns an iterator with the prefix increment (`++`) and dereference (`*`) operators:

```cpp
  for (double value : list) { 
    cout << value << " "; 
  } 
  cout << endl; 
```

We can also iterate through the list manually with the `begin` and `end` methods of the `Iterator` class:

```cpp
  for (LinkedList<double>::Iterator iterator = list.begin(); 
       iterator != list.end(); ++iterator) { 
    cout << *iterator << " "; 
  } 
  cout << endl; 
```

With the `rbegin` and `rend` methods and the `ReverseIterator` class we iterate from its end to its beginning. Note that we still use increment (`++`) rather than decrement (`--`), even though we iterate through the list backwards:

```cpp
  for (LinkedList<double>::ReverseIterator iterator = 
       list.rbegin(); iterator != list.rend(); ++iterator) { 
    cout << *iterator << " "; 
  } 
  cout << endl; 
} 
```

# The extended Set class

The `Set` class of this section has been extended in three ways compared to the version of the earlier section:

*   The set is stored as an ordered list, which makes some of the methods more efficient
*   The class is a template; it may store values of arbitrary types as long as those types support ordering
*   The class has operator overloading, which (hopefully) makes it easier and more intuitive to use

In C++ it is possible to define our own types with the `typedef` keyword. We define `Iterator` of `Set` to be the same iterator as in `LinkedList`. In the earlier section, `Iterator` was a free-standing class that we could reuse when working with sets. However, in this section, `Iterator` is an inner class. Otherwise, `LinkedList` could not be accessed when handling sets since `Set` inherits `LinkedList` privately. Remember that when we inherit privately, all methods and fields of the base class become private in the subclass.

**Set.h:**

```cpp
template <class T> 
class Set : private LinkedList<T> { 
  public: 
    typedef LinkedList<T>::Iterator Iterator; 

    Set(); 
    Set(const T& value); 
    Set(const Set& set); 
    Set& operator=(const Set& set); 
    ~Set(); 
```

We replace the `equal` and `notEqual` methods with overloaded operators for comparison. In this way, it is possible to compare two sets in the same way as when comparing, for instance, two integers:

```cpp
    bool operator==(const Set& set) const; 
    bool operator!=(const Set& set) const; 

    int size() const { return LinkedList<T>::size(); } 
    bool empty() const { return LinkedList<T>::empty(); } 
    Iterator begin() const { return LinkedList<T>::begin(); } 
```

We replace the `unionSet`, `intersection`, and `difference` methods with the operators for addition, multiplication, and subtraction:

```cpp
    Set operator+(const Set& set) const; 
    Set operator*(const Set& set) const; 
    Set operator-(const Set& set) const; 
```

The `merge` function is called by the set methods to perform efficient merging of sets. Since it is a function rather than a method, it must have its own template marking:

```cpp
  private: 
    template <class U> 
    friend Set<U> 
      merge(const Set<U>& leftSet, const Set<U>& rightSet, 
            bool addLeft, bool addEqual, bool addRight); 

  public: 
    Set& operator+=(const Set& set); 
    Set& operator*=(const Set& set); 
    Set& operator-=(const Set& set); 
```

Similar to the preceding `LinkedList` class, we replace the `read` and `write` methods with overloaded stream operators. Since they also are functions rather than methods, they also need their own template markings:

```cpp
    template <class U> 
    friend istream& operator>>(istream& inStream, Set<U>& set); 

    template <class U> 
    friend ostream& operator<<(ostream& outStream, 
                               const Set<U>& set); 
}; 
```

The constructors look pretty much the same, compared to the non-template versions:

```cpp
template <class T> 
Set<T>::Set() { 
  // Empty. 
} 

template <class T> 
Set<T>::Set(const T& value) { 
  add(value); 
} 

template <class T> 
Set<T>::Set(const Set& set) 
 :LinkedList(set) { 
  // Empty. 
} 

template <class T> 
Set<T>::~Set() { 
  // Empty. 
} 

template <class T> 
Set<T>& Set<T>::operator=(const Set& set) { 
  clear(); 
  add(set); 
  return *this; 
} 
```

When testing whether two sets are equal, we can just simply call the equality operator in `LinkedList` since the sets of this section are ordered:

```cpp
template <class T> 
bool Set<T>::operator==(const Set& set) const { 
  return LinkedList::operator==(set); 
} 
```

Similar to the earlier classes, we test whether two sets are not equal by calling `equal`. However, in this class, we use the equality operator explicitly by comparing the own object (by using the `this` pointer) with the given set:

```cpp
template <class T> 
bool Set<T>::operator!=(const Set& set) const { 
  return !(*this == set); 
} 
```

# Union, intersection, and difference

We replace the `unionSet`, `intersection`, and `difference` methods with the addition, subtraction, and multiplication operators. They all call `merge`, with the sets and different values for the `addLeft`, `addEqual`, and `addRight` parameters. In case of union, all three of them are `true`, which means that values present in the left-hand set only, or in both sets, or in the right-hand set only shall be included in the union set:

```cpp
template <class T> 
Set<T> Set<T>::operator+(const Set& set) const { 
  return merge(*this, set, true, true, true); 
} 
```

In case of intersection, only `addEqual` is `true`, which means that the values present in both sets, but not values present in only one of the sets, shall be included in the intersection set. Take a look at the following example:

```cpp
template <class T> 
Set<T> Set<T>::operator*(const Set& set) const { 
  return merge(*this, set, false, true, false); 
} 
```

In case of difference, only `addLeft` is true, which means that only the values present in the left-hand set, but not in both the sets or the right-hand set only, shall be included in the difference set:

```cpp
template <class T> 
Set<T> Set<T>::operator-(const Set& set) const { 
  return merge(*this, set, true, false, false); 
} 
```

The `merge` method takes two sets and the three Boolean values `addLeft`, `addEqual`, and `addRight`. If `addLeft` is true, values present in the left-hand set only are added to the resulting set, if `addEqual` is true, values present in both sets are added, and if `rightAdd` is `true`, values present in the right-hand set only are added:

```cpp
template <class T> 
Set<T> merge(const Set<T>& leftSet, const Set<T>& rightSet, 
             bool addLeft, bool addEqual, bool addRight) { 
  Set<T> result; 
  Set<T>::Iterator leftIterator = leftSet.begin(), 
                   rightIterator = rightSet.begin(); 
```

The `while` statement keeps iterating while there are values left in both the left-hand set and right-hand set:

```cpp
  while ((leftIterator != leftSet.end()) && 
         (rightIterator != rightSet.end())) { 
```

If the left-hand value is smaller, it is added to the resulting set if `addLeft` is `true`. Then the iterator for the left-hand set is incremented:

```cpp
    if (*leftIterator < *rightIterator) { 
      if (addLeft) { 
        result.add(*leftIterator); 
      } 

      ++leftIterator; 
    } 
```

If the right-hand value is smaller, it is added to the resulting set if `addRight` is `true`. Then the iterator for the right-hand set is incremented:

```cpp
    else if (*leftIterator > *rightIterator) { 
      if (addRight) { 
        result.add(*rightIterator); 
      } 

      ++rightIterator; 
    } 
```

Finally, if the values are equal, one of them (but not both, since there are no duplicates in a set) is added and both iterators are incremented:

```cpp
    else { 
      if (addEqual) { 
        result.add(*leftIterator); 
      } 

      ++leftIterator; 
      ++rightIterator; 
    } 
  } 
```

If `addLeft` is `true`, all remaining values of the left-hand set, if any, are added to the resulting set:

```cpp
  if (addLeft) { 
    while (leftIterator != leftSet.end()) { 
      result.add(*leftIterator); 
      ++leftIterator; 
    } 
  } 
```

If `addRight` is `true`, all remaining values of the right-hand set, if any, are added to the resulting set:

```cpp
  if (addRight) { 
    while (rightIterator != rightSet.end()) { 
      result.add(*rightIterator); 
      ++rightIterator; 
    } 
  } 
```

Finally, the resulting set is returned using the following:

```cpp
  return result; 
} 
```

When performing the union operator to this set and another set, we simply call the addition operator. Note that we return our own object by using the `this` pointer:

```cpp
template <class T> 
Set<T>& Set<T>::operator+=(const Set& set) { 
 *this = *this + set; 
  return *this; 
} 
```

In the same way, we call the multiplication and subtraction operators when performing intersection and difference on this set and another set. Look at the following example:

```cpp
template <class T> 
Set<T>& Set<T>::operator*=(const Set& set) { 
 *this = *this * set; 
  return *this; 
} 

template <class T> 
Set<T>& Set<T>::operator-=(const Set& set) { 
 *this = *this - set; 
  return *this; 
} 
```

When reading a set, the number of values of the set is input, and then the values themselves are input. This function is very similar to its counterpart in the `LinkedList` class. However, in order to avoid duplicates, we call the compound addition operator (`+=`) instead of the `add` method:

```cpp
template <class T> 
istream& operator>>(istream& inStream, Set<T>& set) { 
  int size; 
  inStream >> size; 

  for (int count = 0; count < size; ++count) { 
    T value; 
    inStream >> value; 
    set += value; 
  } 

  return inStream; 
} 
```

When writing a set we enclose the value in brackets ("`{`" and "`}`") instead of squares ("`[`" and "`]`"), as in the list case:

```cpp
template <class T> 
ostream& operator<<(ostream& outStream, const Set<T>& set) { 
  outStream << "{"; 
  bool first = true; 

  for (const T& value : set) { 
    outStream << (first ? "" : ",") << value; 
    first = false; 
  } 

  outStream << "}"; 
  return outStream; 
} 
```

We test the set by letting the user input two sets, which we iterate manually with iterators and automatically with the `for` statement. We also evaluate the union, intersection, and difference between the sets.

**Main.cpp:**

```cpp
#include <IOStream> 
using namespace std; 

#include "..\ListAdvanced\List.h" 
#include "Set.h" 

void main() { 
  Set<double> s, t; 
  cin >> s >> t; 

  cout << endl << "s: " << s << endl; 
  cout << "t: " << t << endl; 

  cout << endl << "s: "; 
  for (double value : s) { 
    cout << value << " "; 
  } 

  cout << endl << "t: "; 
  for (Set<double>::Iterator iterator = t.begin(); 
       iterator != t.end(); ++iterator) { 
    cout << *iterator << " "; 
  } 

  cout << endl << endl << "union: " << (s + t) << endl; 
  cout << "intersection: " << (s *t) << endl; 
  cout << "difference: " << (s - t) << endl << endl; 
} 
```

When we execute the program, the output is displayed in a command window:

![](img/85e15c8f-935f-43a7-b872-d7cde0f81a6a.png)

# Advanced searching and sorting

We looked at linear search in the earlier section. In this section, we will look at binary search. The binary search algorithm looks for the value in the middle of the list, and then performs the search with half of the list. In this way, it has *O(log[2]n) *since it splits the list in half in each iteration.

**Search.h:**

```cpp
template <class ListType, class ValueType> 
int binarySearch(const ValueType& value, const ListType& list) { 
  ListType::Iterator* positionBuffer = 
    new ListType::Iterator[list.size()]; 

  int index = 0; 
  for (ListType::Iterator position = list.begin(); 
       position != list.end(); ++position) { 
    positionBuffer[index++] = position; 
  } 

  int minIndex = 0, maxIndex = list.size() - 1; 

  while (minIndex <= maxIndex) { 
    int middleIndex = (maxIndex + minIndex) / 2; 
    ListType::Iterator iterator = positionBuffer[middleIndex]; 
    const ValueType& middleValue = *iterator; 

    if (value == middleValue) { 
      return middleIndex; 
    } 
    else if (value < middleValue) { 
      maxIndex = middleIndex - 1; 
    } 
    else { 
      minIndex = middleIndex + 1; 
    } 
  } 

  return -1; 
} 
```

# The merge sort algorithm

The merge sort algorithm divides the list into two equal sublists, sorts the sublists by recursive calls (a recursive call occurs when a method or function calls itself), and then merges the sorted sublist in a way similar to the `merge` method of the extended version of the `Set` class in the earlier section.

**Sort.h:**

```cpp
template <class ListType, class ValueType> 
void mergeSort(ListType& list) { 
  int size = list.size(); 

  if (size > 1) { 
    int middle = list.size() / 2; 
    ListType::Iterator iterator = list.begin(); 

    ListType leftList; 
    for (int count = 0; count < middle; ++count) { 
      leftList.add(*iterator); 
      ++iterator; 
    } 

    ListType rightList; 
    for (; iterator != list.end(); ++iterator) { 
      rightList.add(*iterator); 
    } 

    mergeSort<ListType, ValueType>(leftList); 
    mergeSort<ListType,ValueType>(rightList); 

    ListType resultList; 
    merge<ListType,ValueType>(leftList, rightList, resultList); 
    list = resultList; 
  } 
} 
```

The `merge` method of this section is reusing the idea of `merge` in the extended `Set` class earlier in this chapter:

```cpp
template <class ListType, class ValueType> 
void merge(ListType& leftList, ListType& rightList, 
           ListType& result) { 
  ListType::Iterator leftPosition = leftList.begin(); 
  ListType::Iterator rightPosition = rightList.begin(); 

  while ((leftPosition != leftList.end()) && 
         (rightPosition != rightList.end())) { 
    if (*leftPosition < *rightPosition) { 
      result.add(*leftPosition); 
      ++leftPosition; 
    } 
    else { 
      result.add(*rightPosition); 
      ++rightPosition; 
    } 
  } 

  while (leftPosition != leftList.end()) { 
    result.add(*leftPosition); 
    ++leftPosition; 
  } 

  while (rightPosition != rightList.end()) { 
    result.add(*rightPosition); 
    ++rightPosition; 
  } 
} 
```

# The quick sort algorithm

The quick sort algorithm selects the first value (called the **pivot value**) and then places all values less than the pivot value in the smaller sublist, and all values greater or equal to the pivot value in the larger sublist. Then the two lists are sorted by recursive calls and then just concatenated together. Let's look at the following example:

```cpp
template <class ListType, class ValueType> 
void quickSort(ListType& list) { 
  if (list.size() > 1) { 
    ListType smaller, larger; 
    ValueType pivotValue = *list.begin(); 

    ListType::Iterator position = list.begin(); 
    ++position; 

    for (;position != list.end(); ++position) { 
      if (*position < pivotValue) { 
        smaller.add(*position); 
      } 
      else { 
        larger.add(*position); 
      } 
    } 

    quickSort<ListType,ValueType>(smaller); 
    quickSort<ListType,ValueType>(larger); 
    list = smaller; 
    list.add(pivotValue); 
    list.add(larger); 
  } 
} 
```

The merge sort algorithm is balanced in a way that it always divides the list into two equal parts and sorts them. The algorithm must iterate through the list once to divide them into two sublists and sorts the sublists. Given a list of values, it must iterate through its *n* values and divide the list *log[2]n* times. Therefore, merge sort *O(n log[2]n)*.

The quick sort algorithm, on the other hand, is, in the worst case (if the list is already sorted), no better than insert, select, or bubble sort: *O(n²)*. However, it is fast in the average case.

# Summary

In this chapter, we have created classes for the abstract datatypes list and set. A list is an ordered structure with a beginning and an end, while a set is an unordered structure.

We started off with rather simple versions where the list had separate classes for the cell and iterator. Then we created a more advanced version where we used templates and operator overloading. We also placed the cell and iterator classes inside the list class. Finally, we introduced overloaded reference methods.

In the same way, we started by creating a rather simple and ineffective version of the set class. Then we created a more advanced version with templates and operator overloading, where we stored the values in order to be able to perform the union, intersection, and difference operations in a more effective way.

Moreover, we have implemented the linear and binary search algorithms. The linear search works on every unordered sequence, but it is rather ineffective. The binary search is more effective, but it only works on ordered sequences.

Finally, we looked into sorting algorithms. We started with the simple but rather ineffective insert, select, and bubble sort algorithms. Then we continued with the more advanced and effective merge and quick sort algorithms.

In the next chapter, we will start to build a library management system.