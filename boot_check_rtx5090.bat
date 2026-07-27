@echo off
REM Compat: ce script est desormais generique et couvre toutes les cartes
REM (RTX 50xx / 40xx / 30xx / 20xx...). Voir boot_check.bat.
call "%~dp0boot_check.bat" %*
