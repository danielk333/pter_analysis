
#standard python
import datetime

#third party imports
import numpy as np

#Pter and pytodotxt
from pter.utils import parse_duration

def calculate_total_activity(tasks, h_per_day):

    def _ok_task(task):
        ok = True
        if task.is_completed:
            ok = False
        if task.creation_date is None:
            ok = False
        if 'due' not in task.attributes:
            ok = False
        if 'estimate' not in task.attributes:
            ok = False
        return ok

    start = []
    end = []
    duration = []
    for task in tasks:
        if not _ok_task(task):
            continue
        start.append(task.creation_date)
        due = task.attributes['due'][0].strip()
        due = due.replace(',','')

        end.append(datetime.date.fromisoformat(due))

        td = parse_duration(task.attributes['estimate'])
        duration.append(td.seconds/3600.0)

    duration = np.array(duration) 
    start = np.array(start)
    end = np.array(end)

    activity = np.empty(duration.shape)
    work_time = np.empty(duration.shape)

    for ind in range(len(work_time)):
        work_time[ind] = np.busday_count(start[ind], end[ind])*h_per_day
        if work_time[ind] == 0:
            work_time[ind] = h_per_day

    activity = duration/work_time
    activity_orig = activity.copy()

    dates = np.arange(datetime.date.today(), end.max())
    total_activity = np.empty(dates.shape)
    for ind, date in enumerate(dates):
        select = np.logical_and(date >= start, date <= end)
        select_inds = np.argwhere(select)
        total_activity[ind] = np.sum(activity[select])
        while total_activity[ind] > 1.0 and ind+1 < len(dates):
            mv = select_inds[np.argmax(end[select_inds])][0]
            if date == end[mv]:
                break
            start[mv] = dates[ind+1]

            select = np.logical_and(date >= start, date <= end)
            select_inds = np.argwhere(select)

            work_time[mv] = np.busday_count(start[mv], end[mv])*h_per_day
            if work_time[mv] == 0:
                work_time[mv] = h_per_day
            activity[mv] = duration[mv]/work_time[mv]
            total_activity[ind] = np.sum(activity[select])


    return dates, total_activity, start, end