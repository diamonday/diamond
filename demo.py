# start the JobManager
from tools import emod
#emod.JobManager().start()

# start the CronDaemon
from tools import cron
#cron.CronDaemon().start()

mj = emod.ManagedJob()
mj.edit_traits()