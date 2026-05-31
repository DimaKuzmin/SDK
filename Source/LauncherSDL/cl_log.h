#pragma	once

void clLog			(const char* msg);
void clMsg			(const char *format, ...);
void Status			(const char *format, ...);
void StatusNoMSG	(const char* format, ...);
void Progress		(const float F);

void ProgressMT		(float F);

void Phase			(const char *phase_name);
void PhasesEnd		();
extern size_t GetHeapMemory();

enum IterationStatus
{
	Skip = 0,
	InProgress,
	Pending,
	Complited,
};

struct IterationPhase
{
	xr_string PhaseName = "";
	u32 elapsed_time = 0;
	u32 remain_time = 0;
	IterationStatus status = InProgress;
	float PhasePersent = 0;
	size_t used_memory = 0;
	xr_string AdditionalData;
};

struct IterationData
{
	xr_string iterationName;

	u32 elapsed_time = 0; // Общее время работы итерации

	int warnings = 0;
	IterationStatus status = Pending;
	xr_vector<IterationPhase> phases;
	float Persent = 0;
};


float					  GetProgress();
xr_vector<IterationData>& GetIterationData();
IterationData* GetActiveIteration();
void SetActiveIteration(IterationData* i);
xr_vector<xr_string>& GetLogVector();
u32&					 GetPhaseStartTime();
xr_string make_time(u32 sec);


void AditionalData(const char* format, ...);
