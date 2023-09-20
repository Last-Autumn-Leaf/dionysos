from ortools.sat.python import cp_model


# Chaque jour est divisé en trois périodes de huit heures.
# Chaque jour, chaque équipe est affectée à un seul employé, et aucun employé ne travaille plusieurs fois.
# Chaque employé est affecté à au moins deux équipes sur trois jours.
def main():
    num_employee = 4
    num_shifts = 3  # 24h/num_shifts
    num_days = 3  # should not be more than 7
    all_employees = range(num_employee)
    all_shifts = range(num_shifts)
    all_days = range(num_days)

    # Creates the model.
    model = cp_model.CpModel()

    # TODO use itertools to optimize this part
    # Creates shift variables.
    # shifts[(n, d, s)]: employee 'n' works shift 's' on day 'd'.
    shifts = {}
    for n in all_employees:
        for d in all_days:
            for s in all_shifts:
                shifts[(n, d, s)] = model.NewBoolVar(f"shift_n{n}_d{d}_s{s}")

    # Each shift is assigned to exactly one employee in the schedule period. (2)
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
    min_shifts_per_employee = (num_shifts * num_days) // num_employee
    if num_shifts * num_days % num_employee == 0:
        max_shifts_per_employee = min_shifts_per_employee
    else:
        max_shifts_per_employee = min_shifts_per_employee + 1

    for n in all_employees:
        shifts_worked = []
        for d in all_days:
            for s in all_shifts:
                shifts_worked.append(shifts[(n, d, s)])
        model.Add(min_shifts_per_employee <= sum(shifts_worked))
        model.Add(sum(shifts_worked) <= max_shifts_per_employee)

    # Creates the solver and solve.
    solver = cp_model.CpSolver()
    solver.parameters.linearization_level = 0
    # Enumerate all solutions.
    solver.parameters.enumerate_all_solutions = True

    class employeesPartialSolutionPrinter(cp_model.CpSolverSolutionCallback):

        def __init__(self, shifts, num_employees, num_days, num_shifts, limit):
            cp_model.CpSolverSolutionCallback.__init__(self)
            self._shifts = shifts
            self._num_employees = num_employees
            self._num_days = num_days
            self._num_shifts = num_shifts
            self._solution_count = 0
            self._solution_limit = limit

        def on_solution_callback(self):
            self._solution_count += 1
            print(f"Solution {self._solution_count}")
            for d in range(self._num_days):
                print(f"Day {d}")
                for n in range(self._num_employees):
                    is_working = False
                    for s in range(self._num_shifts):
                        if self.Value(self._shifts[(n, d, s)]):
                            is_working = True
                            print(f"  employee {n} works shift {s}")
                    if not is_working:
                        print(f"  employee {n} does not work")
            if self._solution_count >= self._solution_limit:
                print(f"Stop search after {self._solution_limit} solutions")
                self.StopSearch()

        def solution_count(self):
            return self._solution_count

    # Display the first five solutions.
    solution_limit = 5
    solution_printer = employeesPartialSolutionPrinter(
        shifts, num_employee, num_days, num_shifts, solution_limit
    )

    solver.Solve(model, solution_printer)

    # Statistics.
    print("\nStatistics")
    print(f"  - conflicts      : {solver.NumConflicts()}")
    print(f"  - branches       : {solver.NumBranches()}")
    print(f"  - wall time      : {solver.WallTime()} s")
    print(f"  - solutions found: {solution_printer.solution_count()}")


if __name__ == "__main__":
    main()
