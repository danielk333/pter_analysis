
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
from pytodotxt import TodoTxt
from pter.utils import parse_duration
from pter.searcher import Searcher

#local import
from .config import get_config, CONFIGFILE
from . import analysis

def prepare(todotxt, config, search):
    cfg = get_config(config)

    if len(todotxt) == 0:
        todotxt = cfg.get('General', 'todotxt-file')
    if len(todotxt) == 0:
        raise ValueError(f'No todotxt file given nor configured in {config}')

    todo = TodoTxt(todotxt)
    todo.parse()

    sch = Searcher(
        search,
        cfg.getboolean('General', 'search-case-sensitive'),
    )

    sel_tasks = [task for task in tasks if sch.match(todo.tasks)]

    return cfg, sel_tasks



@click.group()
@click.version_option()
def cli():
    pass


@cli.command()
@click.option('--search', default='', type=str, help='Pter type search string')
@click.option('--config', default=CONFIGFILE, help='Path to config file')
@click.argument('todotxt', default='', type=str, required=False, help='Path to todotxt file file')
def burndown(todotxt, config, search):
    '''Creates a burndown chart for the selection.
    '''

    cfg, tasks = prepare(todotxt, config, search)

    nominal_day = parse_duration(cfg.get('General','work-day-length'))
    nominal_week = parse_duration(cfg.get('General','work-week-length'))

    h_per_day = nominal_week.days*(nominal_day.seconds/3600.0)/5.0

    dates, total_activity, start, end = analysis.calculate_total_activity(tasks, h_per_day)

    if cfg.getboolean('General', 'usetex'):
        plt.rc('text', usetex=True)


    fig, ax = plt.subplots(1, 1, constrained_layout=True, figsize=(6, 6))
    locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)

    for ind in range(len(start)):
        ax.axvline(start[ind], color='g', alpha=0.1)
        ax.axvline(end[ind], color='r', alpha=0.1)
    max_activity = 100

    activity_label = ''
    if len(context) > 0:
        activity_label += f'@{context} '
    if len(project) > 0:
        activity_label += f'+{project}'

    if ref_activity is None:
        activity_label = 'Total activity'
    else:
        if ref_activity.max()*100 > max_activity:
            max_activity = ref_activity.max()*100
        ax.plot(ref_dates, ref_activity*100, color='k', label='Total activity')

    if total_activity.max()*100 > max_activity:
            max_activity = total_activity.max()*100
    ax.plot(dates, total_activity*100, color='m', label=activity_label)
    ax.set_title('Nominal workload task burn-down',fontsize=24)
    ax.set_ylabel('Full-time workload [\%]',fontsize=20)
    ax.set_ylim([0,max_activity])
    ax.legend(fontsize=16)

    plt.show()
