// xrAI.cpp : Defines the entry point for the application.
//

#include "stdafx.h"
#include "process.h"

#include "../xrlclight/xrlc_light.h"
//#pragma comment(linker,"/STACK:0x800000,0x400000")




extern void logThread			(void *dummy);
extern volatile BOOL bClose;

static const char* h_str = 
	"The following keys are supported / required:\n"
	"-? or -h	== this help\n"
	"-f<NAME>	== compile level in gamedata\\levels\\<NAME>\\\n"
	"-o			== modify build options\n"
	"\n"
	"NOTE: The last key is required for any functionality\n";

void Help()
{	MessageBox(0,h_str,"Command line options",MB_OK|MB_ICONINFORMATION); }
 
void Startup(LPSTR     lpCmdLine)
{

}

int APIENTRY WinMain(HINSTANCE hInstance,
                     HINSTANCE hPrevInstance,
                     LPSTR     lpCmdLine,
                     int       nCmdShow)
{
	// Initialize debugging
	Debug._initialize	(false);
	Core._initialize	("xrDO");
	Startup				(lpCmdLine);
	
	return 0;
}
