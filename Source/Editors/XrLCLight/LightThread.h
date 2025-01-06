#ifndef __LIGHTTHREAD_H__
#define __LIGHTTHREAD_H__


#include "xrthread.h"
#include "detail_slot_calculate.h"

class	LightThread : public CThread
{
	DWORDVec	box_result;
public:
	LightThread			(u32 ID) : CThread(ID)
	{
 	}

	virtual void		Execute();

};
#endif //__LIGHTTHREAD_H__