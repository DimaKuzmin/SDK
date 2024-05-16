#ifndef	_XRLIGHT_IMPLICIED_H_
#define	_XRLIGHT_IMPLICIED_H_

class INetReader ;
class net_task_callback;
class ImplicitDeflector;
#include "../../xrcdb/xrcdb.h"

#include "base_color.h"
#include "base_lighting.h"
#include "xrFaceDefs.h"

#include "base_face.h"
 

class ImplicitExecute
{
	// Data for this thread
	int TH_ID;

public:
 
	ImplicitExecute(int ID) : TH_ID(ID)
	{
	}

	ImplicitExecute() 
	{

	}

protected:
	CDB::COLLIDER DB;
 	
	u32 Jcount;
	Fvector2* Jitter;
	Fvector2 dim;
	Fvector2 half; 
	Fvector2 JS;
public:


	void		Execute			();

 	void		ForCycle		(ImplicitDeflector* defl, u32 V, int TH);
	void		clear();
};


#endif