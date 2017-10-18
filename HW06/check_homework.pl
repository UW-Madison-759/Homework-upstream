use strict;
use warnings;

use Scalar::Util qw(looks_like_number);
use File::Copy qw(copy);
use Cwd qw(cwd);

#----------------------------------------------------------------------------------------------------------------
sub execute($%) {
	my ($cmd, %args) = @_;
	my $silent = $args{'silent'} ? ' 2>&1 >/dev/null' : '';
	my $ret = `$cmd $silent`;
	return $ret, !(( $? >> 8 ) != 0 || $? == -1 || ( $? & 127 ) != 0), $?;
}
sub try_build($$) {
	my ($dir, $target) = @_;
	execute("cd $dir; make $target", silent=>1);
	return 1 if -f "$dir/$target";
	execute("cd $dir; make", silent=>1);
	return 1 if -f "$dir/$target";
	return 0;
}
sub detect_build_system($) {
	my $dir = shift;
	if(-f "$dir/CMakeLists.txt") {
		unlink "$dir/CMakeCache.txt", "cmake_install.cmake";
		use File::Path qw(remove_tree);
		remove_tree("$dir/CMakeFiles");
		my (undef, $pass) = execute("cd $dir; cmake CMakeLists.txt", silent=>1);
		return $pass;
	}
	return 1 if (-f "$dir/Makefile" || -f "$dir/makefile");
	return 0;
}
sub list_any(&@) {
	my $fn = \&{shift @_};
	for (@_) {
		return 1 if $fn->($_);
	}
	return 0;
}
sub list_equals($$) {
	my ($l1, $l2) = @_;
	return 0 unless @$l1 == @$l2; # same length
	for my $i (0..@$l1-1) {
		return 0 unless $l1->[$i] == $l2->[$i];
	}
	return 1;
}
sub clean_output($) {
	my ($out, $delim) = shift;
	$delim //= ' ';
	$out =~ s/\n/$delim/g; $out =~ s/\t/$delim/g;
	$out =~ s/$delim$delim/$delim/g;
	$out =~ s/\[//g; $out =~ s/\]//g;
	$out =~ s/\(//g; $out =~ s/\)//g;
	$out =~ s/\,/ /g; $out =~ s/\:/ /g;
	return split($delim, $out);
}
#----------------------------------------------------------------------------------------------------------------

my $dir = cwd();

$\ = "\n";

# Is there some kind of build system?
if (!&detect_build_system($dir)) {
	print "No build system found";
}

my $problem1_solution = 15.1672;
my @problem2_solution = (0..7, 1..8);

sub problem1($$$) {
	my ($suffix, $dir, $N) = @_;

	my ($out, $pass, $status) = execute("cd $dir; ./problem1$suffix $N");
	if (!$pass) {
		print "Problem 1$suffix with N=$N FAILED: problem1$suffix execution failed (status $status)";
	} else {
		my @res = clean_output($out);
		if (@res != 3) {
			print "Problem 1$suffix with N=$N FAILED. Expected 3 outputs, got '@res' +1";
		} else {
			if (list_any {!looks_like_number $_;} @res) {
				print "Problem 1$suffix with N=$N FAILED. Expected 3 numeric results, got '@res' +2";
			} else {
				if (abs($res[0] - $problem1_solution) > 0.05) {
					print "Problem 1$suffix with N=$N FAILED. Got '$res[0]', expected '$problem1_solution' +3";
				} else {
					print "Problem 1$suffix with N=$N PASSED +4";
				}
			}
		}
	}
}

sub problem3($$$) {
	my ($dir, $params, $solution) = @_;

	my ($out, $pass, $status) = execute("cd $dir; ./problem3 @{$params}");
	if (!$pass) {
		print "Problem 3 FAILED: problem3 execution failed (status $status)";
	} else {
		my @res = clean_output($out);
		if (@res != 5) {
			print "Problem 3 FAILED. Expected 5 outputs, got @{[scalar @res]} +1";
		} else {
			if (list_any {!looks_like_number $_;} @res) {
				print "Problem 3 FAILED. Expected 5 numeric results, got '@res' +2";
			} else {
				if (abs($res[4] - $solution) > 0.05) {
					print "Problem 3 FAILED. Got '$res[4]', expected $solution +3";
				} else {
					print "Problem 3 PASSED +4";
				}
			}
		}
	}
}

# Is there some kind of build system?
if (!detect_build_system($dir)) {
	print "No build system found";
}

if (!try_build($dir, "problem1A")) {
	print "problem1A not found";
} else {
	print "problem1A built successfully +1";
	problem1('A', $dir, 1);
}		
if (!-f "$dir/problem1A.pdf") {
	print "Problem 1 FAILED: problem1A.pdf not found";
} else {
	print "found problem1A.pdf, Problem 1 PASSED +1";
}

if (try_build($dir, "problem1B")) {
	problem1('B', $dir, 1);

	if (!-f "$dir/problem1B.pdf") {
		print "Problem 1 FAILED: problem1B.pdf not found";
	} else {
		print "found problem1B.pdf, Problem 1 PASSED +1";
	}
}

# Problem 2
if (!try_build($dir, "problem2")) {
	print "problem2 not found";
} else {
	print "problem2 built successfully +1";
	my ($out, $pass, $status) = execute("cd $dir; ./problem2");
	if (!$pass) {
		print "Problem 2 FAILED: problem2 execution failed (status $status)";
	} else {
		my @res = clean_output($out);
		if (@res != 16) {
			print "Problem 2 FAILED. Expected 16 outputs, got @{[scalar @res]} +1";
		} else {
			if (list_any {!looks_like_number $_;} @res) {
				print "Problem 2 FAILED. Expected 16 numeric results, got '@res' +2";
			} else {
				if (!list_equals(\@res, \@problem2_solution)) {
					print "Problem 2 FAILED. Got '@res', expected '@problem2_solution' +3";
				} else {
					print "Problem 2 PASSED +4";
				}
			}
		}
	}
}

# Problem 3
if (!try_build($dir, "problem3")) {
	print "problem3 not found";
} else {
	print "problem3 built successfully +1";
	problem3($dir, [1024, 128], 0.53);
}

if (!-f "$dir/problem3.pdf") {
	print "Problem 3 FAILED: problem3.pdf not found";
} else {
	print "found problem3.pdf, Problem 3 PASSED +1";
}
