#include "stdafx.h"
#include "xrDebug.h"

#include <DbgHelp.h>
#include <windows.h>
#pragma comment(lib, "Dbghelp.lib")
#pragma comment(lib, "Kernel32.lib")

typedef USHORT(WINAPI* CaptureStackBackTraceType)(__in ULONG, __in ULONG, __out PVOID*, __out_opt PULONG);
#pragma warning(disable: 4995)


void  callstack_mdmp(_EXCEPTION_POINTERS* pExceptionInfo)
{
	IWriter* writer_callstack = 0;

	//if (writer_callstack == 0)
	{
 		string_path pp;
		if (FS.path_exist("$logs$"))
			FS.update_path(pp, "$logs$", "callstack_dump.cpp");
		writer_callstack = FS.w_open(pp);

		SymSetOptions(SYMOPT_UNDNAME | SYMOPT_DEFERRED_LOADS);
		if (!SymInitialize(GetCurrentProcess(), NULL, TRUE))
			writer_callstack->w_string("Symbols Not Initilized!!!");
		else
			writer_callstack->w_string("Symbols Initilized!!!");
	}

	writer_callstack->w_string("Start Callstack: ");
	writer_callstack->flush();

	if (pExceptionInfo)
	{
		writer_callstack->w_string("Initialize Ctxt: ");
		auto context = pExceptionInfo->ContextRecord;

		HANDLE process = GetCurrentProcess();
		HANDLE thread = GetCurrentThread();

		STACKFRAME64 stackFrame = { 0 };
		DWORD machineType = IMAGE_FILE_MACHINE_I386; // assuming x86 architecture
		DWORD framePtr = 0;

#ifdef _WIN64
		machineType = IMAGE_FILE_MACHINE_AMD64;
#endif

		stackFrame.AddrPC.Mode = AddrModeFlat;
		stackFrame.AddrFrame.Mode = AddrModeFlat;
		stackFrame.AddrStack.Mode = AddrModeFlat;

#ifdef _WIN64
		stackFrame.AddrPC.Offset = context->Rip;
		stackFrame.AddrFrame.Offset = context->Rbp;
		stackFrame.AddrStack.Offset = context->Rsp;
#else
		stackFrame.AddrPC.Offset = context->Eip;
		stackFrame.AddrFrame.Offset = context->Ebp;
		stackFrame.AddrStack.Offset = context->Esp;
#endif
		writer_callstack->w_string("Start Reading PTRs Data: ");

		int ID = 0;
		while (StackWalk64(machineType, process, thread, &stackFrame, context, NULL, SymFunctionTableAccess64, SymGetModuleBase64, NULL))
		{
			try
			{
				// Print call stack information
				// You can also use SymGetSymFromAddr64 to get symbol information
				//std::cout << "Address: " << std::hex << stackFrame.AddrPC.Offset << std::endl;

				IMAGEHLP_SYMBOL64* symbol_x64 = (IMAGEHLP_SYMBOL64*)malloc(sizeof(IMAGEHLP_SYMBOL64) + MAX_SYM_NAME);
				//IMAGEHLP_SYMBOL64 symbol_x64;

				symbol_x64->SizeOfStruct = sizeof(IMAGEHLP_SYMBOL64);
				symbol_x64->MaxNameLength = MAX_SYM_NAME;

				DWORD64 displacementSym;
				SymGetSymFromAddr64(process, stackFrame.AddrPC.Offset, &displacementSym, symbol_x64);

				IMAGEHLP_LINE64 symbol_data;
				DWORD dwDisplacement;
				SymGetLineFromAddr(process, stackFrame.AddrPC.Offset, &dwDisplacement, &symbol_data);

				DWORD_PTR displacement = 0;
				const DWORD max_name_len = 2000;
				char nameBuffer[max_name_len];
				UnDecorateSymbolName(symbol_x64->Name, nameBuffer, max_name_len, UNDNAME_COMPLETE);


				string256 tmp;
				sprintf(tmp, "callstack[%d]: File: %s, Line: %d, SYMVOL:%s", ID, symbol_data.FileName, symbol_data.LineNumber, nameBuffer);

				writer_callstack->w_string(tmp);
				writer_callstack->flush();
				ID++;
			}
			catch (...)
			{
				Msg("Cant Write Callstack[%d]", ID);
			}

		}

	}
	else
	{
		struct callstack_data
		{
			shared_str name_symvol;
			u32 line;
			shared_str name_file;
		};

		std::vector<callstack_data> Names;

		CaptureStackBackTraceType func_callstack = (CaptureStackBackTraceType)(GetProcAddress(LoadLibrary("kernel32.dll"), "RtlCaptureStackBackTrace"));

		if (func_callstack != nullptr)
		{
			const int max_frames = 64;
			void* callstack[max_frames];

			try
			{
				USHORT frames = func_callstack(0, max_frames, callstack, NULL);

				writer_callstack->w_string("Callstack Try Write");

				SYMBOL_INFO* symbol = (SYMBOL_INFO*)calloc(sizeof(SYMBOL_INFO) + 256 * sizeof(char), 1);

				if (symbol != 0)
				{
					symbol->MaxNameLen = 255;
					symbol->SizeOfStruct = sizeof(SYMBOL_INFO);


					for (USHORT i = 0; i < frames; ++i)
					{

						SymFromAddr(GetCurrentProcess(), (DWORD64)(callstack[i]), 0, symbol);

						IMAGEHLP_LINE64 data;

						DWORD dwDisplacement;
						SymGetLineFromAddr(GetCurrentProcess(), (DWORD64)(callstack[i]), &dwDisplacement, &data);

						DWORD_PTR displacement = 0;
						const DWORD max_name_len = 1024;
						char nameBuffer[max_name_len];
						UnDecorateSymbolName(symbol->Name, nameBuffer, max_name_len, UNDNAME_COMPLETE);

						callstack_data d;
						d.name_symvol = nameBuffer;
						d.line = data.LineNumber;
						d.name_file = data.FileName;
						Names.push_back(d);
						//result << i << ": " << symbol->Name << std::endl;

					}

					free(symbol);
				}
			}
			catch (...)
			{
				writer_callstack->w_string("GLOBAL : EXCEPTION_EXECUTE_HANDLER");
			}
		}
		else
		{
			writer_callstack->w_string("--- CALLSTACK CANT FIND FUNCTION!");
		}
		int id = 0;
		for (auto stack : Names)
		{
			id++;
			string256 tmp;
			sprintf(tmp, "callstack[%d]: SYMVOL:%s, LINE: %d, File: %s", id, stack.name_symvol.c_str(), stack.line, stack.name_file.c_str());

			writer_callstack->w_string(tmp);
			Msg("callstack[%d]: SYMVOL:%s, LINE: %d, File: %s", id, stack.name_symvol.c_str(), stack.line, stack.name_file.c_str());
		}
	}

	writer_callstack->w_string("Stop Callstack");
	writer_callstack->w_string("\n");

	FS.w_close(writer_callstack);

}