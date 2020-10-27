
#standard python
import datetime

#third party imports
import numpy as np

#Pter and pytodotxt
from pter.utils import parse_duration

def calculate_total_activity(tasks, h_per_day, default_estimate=0, end_date=None):

    def _ok_task(task):
        ok = True
        if task.is_completed:
            ok = False
        if task.creation_date is None:
            ok = False
        if not ('due' in task.attributes or 't' in task.attributes):
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
        if 't' in task.attributes:
            due = task.attributes['t'][0].strip()
        else:
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

    if end_date is not None:
        dates = np.arange(datetime.date.today(), end_date)
    else:
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

def group_projects(tasks):
    task_distribution = dict()
    for task in tasks:
        for project in task.projects:
            if project not in task_distribution:
                task_distribution[project] = [task]
            else:
                task_distribution[project] += [task]
    return task_distribution


def get_delays(tasks):
    delays = []
    for task in tasks:
        if 'due' not in task.attributes or 't' not in task.attributes:
            continue
        due = task.attributes['due'][0].strip()
        due = due.replace(',','')
        due = datetime.date.fromisoformat(due)

        t = task.attributes['t'][0].strip()
        t = t.replace(',','')
        t = datetime.date.fromisoformat(t)
        dt = t - due
        delays.append(dt.days)
    return delays


def get_ages(tasks):
    ages = []
    for task in tasks:
        if task.creation_date is None or not task.is_completed:
            continue
        dt = task.completion_date - task.creation_date
        ages.append(dt.days)
    return ages


def distribute_projects(todo, default_estimate=0):

    task_distribution = group_projects(todo.tasks)

    spent_distribution = {key:0 for key in task_distribution}
    estimate_distribution = {key:0 for key in task_distribution}
    num_distribution = {key:0 for key in task_distribution}
    for key, tasks in task_distribution.items():
        for task in tasks:
            if not task.is_completed:
                num_distribution[key] += 1

            try:
                if 'estimate' in task.attributes and not task.is_completed:
                    td_est = parse_duration(task.attributes['estimate'][0])
                    estimate_distribution[key] += td_est.seconds/3600.0
                else:
                    if default_estimate != 0:
                        estimate_distribution[key] += default_estimate
            except:
                pass

            try:
                if 'spent' in task.attributes:
                    td_est = parse_duration(task.attributes['spent'][0].replace('min','m'))
                    spent_distribution[key] += td_est.seconds/3600.0
            except:
                pass

    spent_distribution = {key:item for key, item in spent_distribution.items() if item > 0}
    estimate_distribution = {key:item for key, item in estimate_distribution.items() if item > 0}
    num_distribution = {key:item for key, item in num_distribution.items() if item > 0}

    return spent_distribution, estimate_distribution, num_distribution


def calculate_error(tasks, default_estimate=0):

    def _ok_task(task):
        ok = True
        if not task.is_completed:
            ok = False
        if 'estimate' not in task.attributes and default_estimate == 0:
            ok = False
        if 'spent' not in task.attributes:
            ok = False
        return ok

    categories = {}

    ok_tasks = [task for task in tasks if _ok_task(task)]

    error = np.zeros((len(ok_tasks),))
    for ti, task in enumerate(ok_tasks):
        try:
            if 'estimate' in task.attributes:
                td_est = parse_duration(task.attributes['estimate'][0])
                est = td_est.seconds/3600.0
            else:
                est = default_estimate

            td_meas = parse_duration(task.attributes['spent'][0])
            meas = td_meas.seconds/3600.0
            error[ti] = meas - est
        except:
            error[ti] = np.nan
        
    error = error[np.logical_not(np.isnan(error))]


    return error
