#ifndef LOGFILE_CONTROL_H
#define LOGFILE_CONTROL_H

#include "main/__global__.h"
#include <QSettings>
#include <QFile>

extern AppSettings g_AppSettings;

class logfile_control
{
public:
    logfile_control();
    ~logfile_control();

    void load_inifile();
    void save_inifile();
};

#endif // LOGFILE_CONTROL_H
