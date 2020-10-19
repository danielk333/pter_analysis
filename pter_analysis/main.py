
#standard python
import sys
import datetime
import calendar

#third party imports
import click
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

#Pter and pytodotxt
from pytodotxt import TodoTxt, Task
from pter.utils import parse_duration
from pter.searcher import Searcher

#local import
from .config import get_config, CONFIGFILE
from . import analysis

def prepare(todotxt, config):
    cfg = get_config(config)

    if len(todotxt) == 0:
        todotxt = cfg.get('General', 'todotxt-file')
    if len(todotxt) == 0:
        raise ValueError(f'No todotxt file given nor configured in {config}')

    todo = TodoTxt(todotxt)
    todo.parse()

    if cfg.getboolean('General', 'usetex'):
        plt.rc('text', usetex=True)

    style = cfg.get('General', 'matplotlib-style')
    if len(style) > 0:
        plt.style.use(style)

    return cfg, todo


def apply_serach(cfg, todo, search):
    sch = Searcher(
        search,
        cfg.getboolean('General', 'search-case-sensitive'),
    )

    sel_tasks = [task for task in todo.tasks if sch.match(task)]
    return sel_tasks


@click.group()
@click.version_option()
def cli():
    pass


@cli.command()
@click.argument('SEARCH', nargs=-1)
@click.option('--config', default=CONFIGFILE, help='Path to config file')
@click.option('--todotxt', default='', type=str, help='Path to the todotxt file to analyse')
@click.option('--default-estimate', default=0, type=float, help='If no "estimate" tag is given, assume this many hours (else exclude those tasks)')
def burndown(config, default_estimate, todotxt, search):
    '''Creates a burn-down chart for the selection.

    SEARCH: Pter-type search strings (multiple searches are given by space separation)
    '''

    cfg, todo = prepare(todotxt, config)
    
    nominal_day = parse_duration(cfg.get('General','work-day-length'))
    nominal_week = parse_duration(cfg.get('General','work-week-length'))

    h_per_day = nominal_week.days*(nominal_day.seconds/3600.0)/5.0


    fig, ax = plt.subplots(1, 1, constrained_layout=True, figsize=(9, 4))
    locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    axv_legend = False

    max_activity = 100

    total_activity = None

    for sch in search:
        tasks = apply_serach(cfg, todo, sch)
        dates, activity, start, end = analysis.calculate_total_activity(tasks, h_per_day, default_estimate=default_estimate)

        for ind in range(len(start)):
            if axv_legend:
                ax.axvline(start[ind], color='g', alpha=0.1)
                ax.axvline(end[ind], color='r', alpha=0.1)
            else:
                ax.axvline(start[ind], color='g', alpha=0.1, label='Task creation')
                ax.axvline(end[ind], color='r', alpha=0.1, label='Task due')
                axv_legend = True
        

        if activity.max()*100 > max_activity:
                max_activity = activity.max()*100
        ax.plot(dates, activity*100, label=sch)

        if total_activity is None:
            total_activity = activity.copy()
        else:
            total_activity += activity

    if len(search) > 1:
        ax.plot(dates, total_activity*100, '--k', label='Total activity')

    ax.set_title('Nominal workload task burn-down')
    ax.set_ylabel('Full-time workload [\%]')
    ax.set_ylim([0,max_activity])
    ax.set_xlim([datetime.datetime.today(),None])
    ax.legend()

    plt.show()


@cli.command()
@click.argument('SEARCH', type=str)
@click.option('--todotxt', default='', type=str, help='Path to the todotxt file to analyse')
@click.option('--task-limit', default=1, type=int, help='Max number of tasks per unique date')
@click.option('--name-limit', default=20, type=int, help='Max number of characters per task')
@click.option('--context/--no-context', default=False, help='Remove project tags from task')
@click.option('--project/--no-project', default=True, help='Remove context tags from task')
@click.option('--config', default=str(CONFIGFILE), help='Path to config file')
def timeline(config, project, context, name_limit, task_limit, todotxt, search):
    '''Creates a timeline chart for the due dates of the selection.

    SEARCH: Pter-type search string
    '''
    cfg, todo = prepare(todotxt, config)
    tasks = apply_serach(cfg, todo, search)
    
    def rem_tag(description, regex):
        while True:
            matches = parse_tags(description, regex)
            if len(matches) == 0:
                break
            for match in matches:
                start, end = match.span()
                description = description[:start] + description[end:]
                break
        return description

    def parse_tags(description, regex):
        matches = []
        if description is None:
            return matches

        for match in regex.finditer(description):
            if match:
                matches.append(match)
            else:
                break
        return matches 

    dates = []
    names = []
    num = []
    for task in tasks:
        if 'due' not in task.attributes:
            continue
        due = task.attributes['due'][0].strip()
        due = due.replace(',','')
        date = datetime.date.fromisoformat(due)

        name = task.description

        if not project:
            name = rem_tag(name, Task.PROJECT_RE)
        if not context:
            name = rem_tag(name, Task.CONTEXT_RE)
        name_short = name[:name_limit]

        if len(name) > name_limit:
            name_short += '...'

        if date in dates:
            ind = dates.index(date)
            num[ind] += 1
            if num[ind] > task_limit:
                names[ind] = [f'{num[ind]} Tasks']
            else:
                names[ind].append(name_short)
        else:
            dates.append(date)
            num.append(1)
            if task_limit == 0:
                names.append([f'1 Tasks'])
            else:
                names.append([name_short])

    if len(dates) == 0:
        raise ValueError('No tasks found with due-dates in search')

    levs = [-5, 5, -3, 3, -1, 1]
    if len(dates) < len(levs):
        levs = levs[:len(dates)]

    levels = np.tile(
        levs, 
        int(np.ceil(len(dates)/len(levs))),
    )[:len(dates)]

    # Create figure and plot a stem plot with the date
    fig, ax = plt.subplots(figsize=(8.8, 4), constrained_layout=True)
    ax.set(title=search)

    markerline, stemline, baseline = ax.stem(dates, levels,
                                             linefmt="C3-", basefmt="k-",
                                             use_line_collection=True)

    plt.setp(markerline, mec="k", mfc="w", zorder=3)

    # Shift the markers to the baseline by replacing the y-data by zeros.
    markerline.set_ydata(np.zeros(len(dates)))
    ax.set_ylim(-7,7)

    # annotate lines
    vert = np.array(['top', 'bottom'])[(levels > 0).astype(int)]
    for d, l, r, va in zip(dates, levels, names, vert):
        for ry, name in enumerate(r):
            ax.annotate(name, xy=(d, l-ry), xytext=(-3, np.sign(l)*3),
                        textcoords="offset points", va=va, ha="right")

    # format xaxis with 4 month intervals
    ax.get_xaxis().set_major_locator(mdates.MonthLocator(interval=1))
    ax.get_xaxis().set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    # remove y axis and spines
    ax.get_yaxis().set_visible(False)
    for spine in ["left", "top", "right"]:
        ax.spines[spine].set_visible(False)

    ax.margins(y=0.1)
    plt.show()