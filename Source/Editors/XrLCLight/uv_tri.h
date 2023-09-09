#include "xrfacedefs.h"
#include "tcf.h"
 
struct XRLC_LIGHT_API UVtri : public _TCF		
{
	Face*	owner;
	void	Serialize(IWriter* w); 
	void	Deserialize(IReader* w); 

	void	read				( INetReader	&r );
	void	write				( IWriter	&w ) const ;
	bool	similar				( const UVtri &uv, float eps = EPS ) const;
};

