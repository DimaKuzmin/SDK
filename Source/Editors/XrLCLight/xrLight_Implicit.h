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
	u32					y_start,y_end;

	int TH_ID;
public:
	ImplicitExecute( u32 _y_start, u32 _y_end , int ID): y_start(_y_start), y_end( _y_end ), TH_ID(ID)
	{
	}
	ImplicitExecute(): y_start( u32(-1) ),y_end( u32(-1) )
	{

	}

protected:
	CDB::COLLIDER DB;
	net_task_callback* net_cb;
	
	u32 Jcount;
	Fvector2* Jitter;
	Fvector2 dim;
	Fvector2 half; 
	Fvector2 JS;
public:


	void		Execute			( net_task_callback *net_callback );

 	void		ForCycle		(ImplicitDeflector* defl, u32 V, int TH);

	void		read			( INetReader	&r );
	void		write			( IWriter	&w ) const ;

	void		receive_result			( INetReader	&r );
	void		send_result				( IWriter	&w ) const ;
	void		clear();
};


#endif