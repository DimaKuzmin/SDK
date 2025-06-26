#ifndef PH_ITEM_LIST_H
#define PH_ITEM_LIST_H
/*
#define DECLARE_PHLIST_ITEM(class_name)			public:\
												class CPHListItem\
												{\
													friend class CPHItemList<class_name>;\
													friend class CPHItemList<class_name>::iterator;\
													class_name* next;\
													class_name** tome;\
												};\
												private:
*/
#define DECLARE_PHLIST_ITEM(class_name)			friend class CPHItemList<class_name>;\
												friend class CPHItemList<class_name>::iterator;\
												class_name* next;\
												class_name** tome;
#define DECLARE_PHSTACK_ITEM(class_name)		DECLARE_PHLIST_ITEM(class_name)\
												friend class CPHItemStack<class_name>;\
												u16 stack_pos;

template<class T>
class CPHItemList
{
	std::list<T*> items;

public:
	using iterator = typename std::list<T*>::iterator;

	CPHItemList() = default;

	u16 count() const {
		return static_cast<u16>(items.size());
	}

	void push_back(T* item) {
		items.push_back(item);
	}

	void move_items(CPHItemList<T>& source_list) {
		items.splice(items.end(), source_list.items);
	}

	void erase(iterator it) {
		items.erase(it);
	}

	void empty() {
		items.clear();
	}

	iterator begin() { return items.begin(); }
	iterator end() { return items.end(); }
};

template<class T>
class CPHItemStack :
	public CPHItemList<T>
{
public:
	void			push_back(T* item)
	{
		item->stack_pos = size;
		CPHItemList<T>::push_back(item);
	}
};

#define DEFINE_PHITEM_LIST(T,N,I)		typedef CPHItemList<T>	N; typedef CPHItemList<T>::iterator I;
#define DEFINE_PHITEM_STACK(T,N,I)		typedef CPHItemStack<T>	N; typedef CPHItemStack<T>::iterator I;
#endif
