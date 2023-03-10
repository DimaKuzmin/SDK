#ifndef GAMETYPE_CHOOSER_INCLUDED
#define GAMETYPE_CHOOSER_INCLUDED

#pragma once

//new
enum EGameIDs {
        eGameIDNoGame                                   = u32(0),
        eGameIDSingle                                   = u32(1) << 0,
        eGameIDDeathmatch                               = u32(1) << 1,
        eGameIDTeamDeathmatch                           = u32(1) << 2,
        eGameIDArtefactHunt                             = u32(1) << 3,
        eGameIDCaptureTheArtefact                       = u32(1) << 4,
        eGameIDDominationZone                           = u32(1) << 5,
        eGameIDTeamDominationZone                       = u32(1) << 6,
        eGameIDCoop                                     = u32(1) << 7,
        eGameIDRoleplay                                 = u32(1) << 8
};

class PropValue;
class PropItem;
DEFINE_VECTOR			(PropItem*,PropItemVec,PropItemIt);

struct GameTypeChooser
{
    Flags16	m_GameType;
#ifndef XRGAME_EXPORTS
		void	FillProp		(LPCSTR pref, PropItemVec& items);
#endif // #ifndef XRGAME_EXPORTS


	bool 	LoadStream		(IReader&F);
	bool 	LoadLTX			(CInifile& ini, LPCSTR sect_name, bool bOldFormat);
	void 	SaveStream		(IWriter&);
	void 	SaveLTX			(CInifile& ini, LPCSTR sect_name);

    void    SetValue(const u16& mask, bool& value) { 
        Msg("set[%d] value [%d]", mask, value);
        m_GameType.set(mask, value);
    };
	void	SetDefaults		()				{m_GameType.one();}
	bool	MatchType		(const u16& mask) const		{return m_GameType.test(mask);};
};

#endif