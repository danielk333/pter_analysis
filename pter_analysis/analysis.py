
#standard python
import datetime

#third party imports
import numpy as np

#Pter and pytodotxt
from pter.utils import parse_duration

def calculate_total_activity(tasks, h_per_day, default_estimate=0):

    def _ok_task(task):
        ok = True
        if task.is_completed:
            ok = False
        if task.creation_date is None:
            ok = False
        if 'due' not in task.attributes:
            ok = False
        return ok

    start = []
    end = []
    duration = []
    for task in tasks:
        if not _ok_task(task):
            continue

        if 'estimate' in task.attributes:
            td = parse_duration(task.attributes['estimate'][0])
            duration.append(td.seconds/3600.0)
        else:
            if default_estimate == 0:
                continue
            else:
                duration.append(default_estimate)

        start.append(task.creation_date)
        due = task.attributes['due'][0].strip()
        due = due.replace(',','')
        due = datetime.date.fromisoformat(due)

        if due < datetime.date.today():
            due = datetime.date.today() + datetime.timedelta(days=1)

        end.append(due)


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



def calculate_error(tasks, split_project=True):

    def _ok_task(task):
        ok = True
        if not task.is_completed:
            ok = False
        if 'spent' not in task.attributes:
            ok = False
        if 'estimate' not in task.attributes:
            ok = False
        return ok

    categories = {}

    for task in tasks:
        if not _ok_task(task):
            continue

        for project in task.projects:
            if project not in categories:
                categories[project] = [task]
            else:
                categories[project] += [task]

    labels = categories.keys()
    errors = []

    for key in labels:
        tmp_err = np.zeros((len(categories[key]),))
        for ti,task in enumerate(categories[key]):
            try:
                td_est = parse_duration(task.attributes['estimate'][0])
                est = td_est.seconds/3600.0
                td_meas = parse_duration(task.attributes['spent'][0])
                meas = td_meas.seconds/3600.0
                tmp_err[ti] = meas - est
            except:
                tmp_err[ti] = np.nan
            
        tmp_err = tmp_err[np.logical_not(np.isnan(tmp_err))]
        errors.append(tmp_err)

    return labels, errors
