#pragma once
class UIWayTool :public UIToolCustom
{
	string_path object_prefix;
	int work_id = 0;
	 
	enum WayWorks
	{
		eWorkNone = 0,
		eWorkWalker = 1,
		eWorkPatrol = 2,
		eWorkGuard = 3,
		eWorkSleeper = 4,
		eWorkAnimpoint = 5,
		eWorkSniper = 6, 
		eWorkSurge = 7
	};

	enum WayWorks_Type
	{
		eWorkWalk = 0,
		eWorkLook = 1
	};

	WayWorks work_selected;
	WayWorks_Type work_selected_type;

public:
	UIWayTool();
	virtual ~UIWayTool();
	virtual void Draw();
	IC bool IsAutoLink()const { return m_AutoLink; }
	IC void SetWayMode(bool mode) { m_WayMode = mode; }
private:
	bool m_WayMode;
	bool m_AutoLink;
};