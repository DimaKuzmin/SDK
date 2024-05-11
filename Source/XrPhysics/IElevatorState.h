#pragma once


class IPhysicsShellHolder;
class IElevatorState
{
public:
	enum Estate
	{
		clbNone = 0,
		clbNearUp,
		clbNearDown,
		clbClimbingUp,
		clbClimbingDown,
		clbDepart,
		clbNoLadder,
		clbNoState
	};

	virtual Estate	State		()							= 0;
	virtual	void	NetRelcase	( IPhysicsShellHolder* O )	= 0;
protected:
	virtual	~IElevatorState() = 0 {}
};