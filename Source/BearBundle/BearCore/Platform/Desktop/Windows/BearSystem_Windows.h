

 void BearSystem::ToClipboard(const bchar8 * text)
{
	 ToClipboard(text, BearString::GetSize(text));
}

 void  BearSystem::ToClipboard(const bchar16 * text)
{
	 ToClipboard(*BearEncoding::ToAnsi(text));
}


 void  BearSystem::ToClipboard(const bchar8 * text, bsize size)
 {
	 if (OpenClipboard(0))
	 {
		 HGLOBAL hgBuffer;
		 bchar8* chBuffer;
		 EmptyClipboard(); //������� �����
		 hgBuffer = GlobalAlloc(GMEM_DDESHARE, size + 1);//�������� ������
		 chBuffer = (char*)GlobalLock(hgBuffer); //��������� ������
		BearString::CopyWithSizeLimit(chBuffer, size + 1, text, size);
		 GlobalUnlock(hgBuffer);//������������ ������
		 SetClipboardData(CF_TEXT, hgBuffer);//�������� ����� � ����� ������
		 CloseClipboard();
	 }
	
 }

  void  BearSystem::ToClipboard(const bchar16 * text, bsize size)
 {
	  ToClipboard(*BearEncoding::ToAnsi(text), size);
 }


  BearString  BearSystem::GetClipboard()
{
	 BearString result;
	 if (OpenClipboard(0))
	 {
		 HANDLE hData = GetClipboardData(CF_TEXT);//��������� ����� �� ������ ������
		 bchar8* chBuffer = (bchar8*)GlobalLock(hData);//��������� ������

#ifdef UNICODE
		 result = BearEncoding::ToUTF16(chBuffer);
#else
		 result.assign(chBuffer);
#endif
		
		 GlobalUnlock(hData);//������������ ������
		 CloseClipboard();//��������� ����� ������
	 }
	 return result;
}
