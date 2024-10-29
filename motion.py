import pandas as pd
import numpy as np
import ortools
import matplotlib.pyplot as plt
from ortools.sat.python import cp_model
from pandas import Timestamp

# Define the scheduling horizon and parameters for a one-week schedule
HORIZON_WEEKS = 1
START_DATE = '2024-10-01'
WORKING_DAYS = "M-F"  # Workdays: Monday to Friday
WORKING_HOURS = (9, 17)  # Working hours: 9 AM to 5 PM

tasks = {
    'Complete Budget Draft': {'time_to_complete': 30, 'due_date': Timestamp('2024-10-02 11:00:00')},
    'Client Meeting Preparation': {'time_to_complete': 45, 'due_date': Timestamp('2024-10-03 09:00:00')},
    'Design Mockup Review': {'time_to_complete': 60, 'due_date': Timestamp('2024-10-04 16:00:00')},
    'Prepare Technical Report': {'time_to_complete': 45, 'due_date': Timestamp('2024-10-03 15:30:00')},
    'Plan Marketing Strategy': {'time_to_complete': 30, 'due_date': Timestamp('2024-10-04 14:00:00')},
    'Review Project Milestones': {'time_to_complete': 15, 'due_date': Timestamp('2024-10-02 14:30:00')},
    'Finalize Design Concepts': {'time_to_complete': 15, 'due_date': Timestamp('2024-10-04 12:15:00')},
    'Update Sales Materials': {'time_to_complete': 30, 'due_date': Timestamp('2024-10-01 17:30:00')},
    'Draft Training Manual': {'time_to_complete': 45, 'due_date': Timestamp('2024-10-03 12:00:00')},
    'Consolidate Performance Review': {'time_to_complete': 60, 'due_date': Timestamp('2024-10-04 19:00:00')}
}

events = {
    'Team Alignment Meeting': {'date_time': Timestamp('2024-10-01 09:00:00'), 'length': 60},
    'Sales Review Session': {'date_time': Timestamp('2024-10-01 15:00:00'), 'length': 90},
    'Product Roadmap Discussion': {'date_time': Timestamp('2024-10-02 10:00:00'), 'length': 120},
    'Technical Team Standup': {'date_time': Timestamp('2024-10-02 08:30:00'), 'length': 30},
    'Client Feedback Call': {'date_time': Timestamp('2024-10-03 11:00:00'), 'length': 45},
    'Marketing Strategy Brainstorm': {'date_time': Timestamp('2024-10-03 14:00:00'), 'length': 60},
    'Engineering Sync-Up': {'date_time': Timestamp('2024-10-04 09:00:00'), 'length': 90},
    'Finance Team Huddle': {'date_time': Timestamp('2024-10-04 11:30:00'), 'length': 30},
    'Operations Review': {'date_time': Timestamp('2024-10-04 14:30:00'), 'length': 60},
    'Customer Strategy Roundtable': {'date_time': Timestamp('2024-10-04 18:00:00'), 'length': 90}
}

# Calculate the end date based on the scheduling horizon in weeks
end_date = pd.to_datetime(START_DATE) + pd.Timedelta(weeks=HORIZON_WEEKS)

# Create a date range with 15-minute increments to represent all possible time slots in the schedule
date_range = pd.date_range(start=START_DATE, end=end_date, freq='15min')

# Create a DataFrame to store the schedule, using the generated date range as the index
df_schedule = pd.DataFrame(index=date_range)

# Add a column to track the day of the week for each time slot
df_schedule['dow'] = df_schedule.index.dayofweek  # Monday=0, Sunday=6

# Create a column to determine if each 15-minute slot falls within working hours and on a working day
df_schedule['working_hour'] = df_schedule.index.to_series().apply(
    lambda x: 1 if (x.weekday() < 5 and WORKING_HOURS[0] <= x.hour < WORKING_HOURS[1]) else 0
)


# Initialize a column in the DataFrame to track time slots that are occupied by existing events
df_schedule['scheduled'] = 0

# Mark time slots occupied by pre-existing events as "scheduled" in the DataFrame
for event, details in events.items():
    start_time = details['date_time']
    duration_minutes = details['length']
    end_time = start_time + pd.Timedelta(minutes=duration_minutes)

    # Mark the event slots as unavailable by setting `scheduled` to 1
    df_schedule.loc[start_time:end_time, 'scheduled'] = 1

# Create a boolean column `available` to identify time slots that are within working hours and not scheduled
df_schedule['available'] = (df_schedule['working_hour'] == 1) & (df_schedule['scheduled'] == 0)

# Create a numerical index to reference the time slots for easy access later
df_schedule['time_index'] = range(len(df_schedule))
available_slots = df_schedule[df_schedule['available']].index

# Initialize the CP-SAT model for scheduling optimization
model = cp_model.CpModel()

# Store `available` status as a list to easily use in constraints
available = df_schedule['available'].tolist()

# Initialize variables for each time slot indicating whether it is occupied
master_timestep_vars = [model.NewIntVar(name=f'scheduled_step_{i}', lb=0, ub=1) for i in range(0, df_schedule.shape[0])]

# Get the maximum index (last time slot) for easier constraints
max_index = df_schedule['time_index'].max()

# Create task-specific variables to track which slots are occupied by each task
increment_vars = [[model.NewBoolVar(f'increment_var_{task_name}_{offset}')
                   for offset in range(max_index + 1)] for task_name in tasks]

task_vars = []
task_start_vars = []
scheduled_vars = []
task_i = 0

# Loop through all tasks to set up variables and constraints for each
for task in tasks:
    details = tasks[task]
    due_time = details['due_date']
    duration = details['time_to_complete'] // 15  # Duration in 15-minute increments
    task_increment_vars = increment_vars[task_i]

    # Find the maximum index up to which the task can be scheduled based on its due date
    max_due_index = df_schedule[df_schedule.index <= due_time]['time_index'].max()

    # Boolean variables to track the task's schedule across all time slots
    task_timesteps = [model.NewBoolVar(f'scheduled_step_{task}_{ts}') for ts in range(0, max_index)]

    # Integer variable to store the start time of the task
    start_timestep = model.NewIntVar(name=f'scheduled_start_step_{task}', ub=max_index, lb=0)

    # Variable to track the count of scheduled slots for the task
    scheduled_itterator = model.NewIntVar(name=f"scheduled_iter_{task}", lb=0, ub=duration + 1)
    # Boolean variable indicating if the task is fully scheduled
    scheduled = model.NewBoolVar(f"{task}_scheduled")

    # Constraint: Once a task starts, its duration slots must be occupied
    for offset in range(duration):
        # Track the start and subsequent slots for the task
        shifted_index = model.NewIntVar(0, max_due_index, f'shifted_index_{task}_{offset}')
        model.Add(shifted_index == start_timestep + offset)
        scheduled_var = model.NewBoolVar(f'Scheduled_{task}_{offset}')

        # Use AddElement to link time slots with task scheduling
        model.AddElement(shifted_index, task_timesteps, scheduled_var)
        model.AddElement(shifted_index, task_increment_vars, scheduled_var)

        # Each `scheduled_var` must be `True` for a valid schedule
        model.Add(scheduled_var == 1)

    # Constraint: `scheduled_iterator` counts the total scheduled time slots for a task
    model.add(scheduled_itterator == sum(task_timesteps))

    # If the scheduled time slots meet the task duration, mark it as `True`
    model.add(scheduled_itterator >= duration).OnlyEnforceIf(scheduled)

    # Track task's start time and scheduling status
    task_start_vars.append(start_timestep)
    scheduled_vars.append(scheduled)

    task_i += 1

# Constraints for ensuring that each slot is not double-booked and follows availability rules
for i in range(0, max_index):
    availability = available[i]
    indexed_increments = []
    for t in increment_vars:
        indexed_increments.append(t[i])

    # Ensure only one task can occupy each slot
    model.Add(master_timestep_vars[i] == sum(indexed_increments))

    # Ensure tasks are only scheduled within available slots
    if availability == False:
        model.Add(master_timestep_vars[i] == 0)

# Objective function: maximize the number of fully scheduled tasks
total_scheduled = model.NewIntVar(name="Total_Scheduled", lb=0, ub=1000)
model.add(total_scheduled == sum(scheduled_vars))
model.Maximize(total_scheduled)

# Solve the model and check for an optimal or feasible solution
solver = cp_model.CpSolver()
status = solver.Solve(model)

# Output the results of the schedule optimization
if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
    print('Solution found:')
    t_i = 0
    for task_name in tasks.keys():
        is_scheduled = solver.Value(scheduled_vars[t_i])
        if is_scheduled:
            start_index = solver.Value(task_start_vars[t_i])
            start_time = df_schedule[df_schedule['time_index'] == start_index].index[0]
            print(f'{task_name} starts at {start_time}')
        else:
            print(f'{task_name} could not be scheduled')
        t_i += 1
else:
    print('No feasible solution found.')
