#ifndef	_XRLIGHT_IMPLICIED_H_
#define	_XRLIGHT_IMPLICIED_H_
class INetReader ;
 
class ImplicitExecute
{
	// Data for this thread
	u32 MAX_HEIGHT = 0;

public:
	ImplicitExecute(u32 MAX_H) : MAX_HEIGHT(MAX_H)
	{
	
	}
	 
	void		Execute			(   );
 
};


#endif