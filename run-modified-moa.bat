@echo off

set MEMORY=5G
set LIBDIR=.\irnd\lib
set DISTDIR=.\irnd\lib
set MOAJAR=%LIBDIR%\moa.jar
set MOASZJAR=%LIBDIR%\sizeofag.jar
set IRNDJAR=%LIBDIR%\..\dist\irnd.jar
set 
set MOACMD=java -Xmx%MEMORY% -cp %IRNDJAR%;%MOAJAR% -javaagent:%MOASZJAR% moa.gui.GUI



%MOACMD% 
	



