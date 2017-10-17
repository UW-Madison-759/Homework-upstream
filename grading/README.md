## Grading Rules

### Get one point for each of the following
	1. Executable successfully builds
	2. Executable successfully runs
	3. Reports correct number of values
	4. Each value has correct type
	5. Content of reported values is correct (from file or stdout)

There are a total of five (5) points per run of each executable. There may be multiple runs with each having a different configuration (e.g., different numbers of threads). Get 1 point if PDF of plots is present (where needed)

###### NOTE: The only way to get 0 points is to have no suitable build system (i.e., no Makefile or CMakeLists.txt).

---

## Failure Cases

Here we list some, but not all, possible reasons for points to be lost on each of the rules above.

### Rule 1 (Executable successfully builds)
	1. Error in the Makefile (e.g., spaces instead of tabs)
	2. Compile or link error
	3. Executable not built due to incorrecty specified dependencies (e.g., executable is not a dependency of 'all')

### Rule 2 (Executable successfully runs)
	1. return non-zero value from 'main'
	2. stack overflow
	3. heap corruption
	4. uncaught exceptions (C++)
	5. Incorrectly named executable
	6. Command line arguements incorrectly handled
	7. Reading from incorrect file (if statically named)

### Rule 3 (Reports correct number and type of values)
	1. outputs are in wrong order
	2. misnamed output files
	3. extraneous output (e.g., 'The sum is 7' rather than '7')
	4. any non-whitespace deviations from the specifications of the problem statement

### Rule 4 (Content of reported values is correct (from file or stdout))
	1. Outputs do not match expected result for the problem
	2. Logic errors
	3. Incorrect algorithm used

Students are strongly encouraged to run their code under the sanitizers or an instrumentation program like Valgrind before submitting their code. This will prevent most of the failure cases for Rule 2.

---

## Partial Credit

Partial credit is given at the grader's discretion. Reasonable effort will be made to ensure that small mistakes do not overly penalize a student's otherwise good work. When considering partial credit, the following guidelines are used

	1. Multiple instances of the same issue are only counted once
		a. Example: Three of the output files have extra lines of text in them. This would only count as a loss of one point, not three.
	2. For Rule 4, partial credit can only be given for 4.3
	3. Items checked by the grading script provided to students before the due date are unlikely to be considered for partial credit
		a. These include file names, output formatting, and build success
