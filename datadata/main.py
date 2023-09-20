from pathlib import Path
from utils import *
import pandas as pd
from datetime import datetime, date
import matplotlib.pyplot as plt

# DFS=initialize()
# poicon=DFS[POICON]

"""employee scheduling problem with shift requests."""
from ortools.sat.python import cp_model


def main():
    # This program tries to find an optimal assignment of employees to shifts
    # (3 shifts per day, for 7 days), subject to some constraints (see below).
    # Each employee can request to be assigned to specific shifts.
    # The optimal assignment maximizes the number of fulfilled shift requests.
    num_employees = 5
    num_shifts = 3
    num_days = 7
    all_employees = range(num_employees)
    all_shifts = range(num_shifts)
    all_days = range(num_days)
    shift_requests = [
        [[0, 0, 1], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 0, 1]],
        [[0, 0, 0], [0, 0, 0], [0, 1, 0], [0, 1, 0], [1, 0, 0], [0, 0, 0], [0, 0, 1]],
        [[0, 1, 0], [0, 1, 0], [0, 0, 0], [1, 0, 0], [0, 0, 0], [0, 1, 0], [0, 0, 0]],
        [[0, 0, 1], [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 0], [1, 0, 0], [0, 0, 0]],
        [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 0]],
    ]

    # shift_requests[i][j] represent the ith employee jth day wants to be on the x shift (1)
    # Creates the model.
    model = cp_model.CpModel()

    # Creates shift variables.
    # shifts[(n, d, s)]: employee 'n' works shift 's' on day 'd'.
    shifts = {}
    # O(x*y*z)
    for n in all_employees:
        for d in all_days:
            for s in all_shifts:
                shifts[(n, d, s)] = model.NewBoolVar(f"shift_n{n}_d{d}_s{s}")

    # Each shift is assigned to exactly one employee in .
    for d in all_days:
        for s in all_shifts:
            model.AddExactlyOne(shifts[(n, d, s)] for n in all_employees)

    # Each employee works at most one shift per day.
    for n in all_employees:
        for d in all_days:
            model.AddAtMostOne(shifts[(n, d, s)] for s in all_shifts)

    # Try to distribute the shifts evenly, so that each employee works
    # min_shifts_per_employee shifts. If this is not possible, because the total
    # number of shifts is not divisible by the number of employees, some employees will
    # be assigned one more shift.
    min_shifts_per_employee = (num_shifts * num_days) // num_employees
    if num_shifts * num_days % num_employees == 0:
        max_shifts_per_employee = min_shifts_per_employee
    else:
        max_shifts_per_employee = min_shifts_per_employee + 1
    for n in all_employees:
        num_shifts_worked = 0
        for d in all_days:
            for s in all_shifts:
                num_shifts_worked += shifts[(n, d, s)]
        model.Add(min_shifts_per_employee <= num_shifts_worked)
        model.Add(num_shifts_worked <= max_shifts_per_employee)

    # pylint: disable=g-complex-comprehension
    model.Maximize(
        sum(
            shift_requests[n][d][s] * shifts[(n, d, s)]
            for n in all_employees
            for d in all_days
            for s in all_shifts
        )
    )

    # Creates the solver and solve.
    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    if status == cp_model.OPTIMAL:
        print("Solution:")
        for d in all_days:
            print("Day", d)
            for n in all_employees:
                for s in all_shifts:
                    if solver.Value(shifts[(n, d, s)]) == 1:
                        if shift_requests[n][d][s] == 1:
                            print("employee", n, "works shift", s, "(requested).")
                        else:
                            print("employee", n, "works shift", s, "(not requested).")
            print()
        print(
            f"Number of shift requests met = {solver.ObjectiveValue()}",
            f"(out of {num_employees * min_shifts_per_employee})",
        )
    else:
        print("No optimal solution found !")

    # Statistics.
    print("\nStatistics")
    print(f"  - conflicts: {solver.NumConflicts()}")
    print(f"  - branches : {solver.NumBranches()}")
    print(f"  - wall time: {solver.WallTime()}s")


if __name__ == "__main__":
    main()
