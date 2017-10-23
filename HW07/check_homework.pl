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
my $problem1_solution = -179;

sub problem1($$) {
	my ($dir, $grade) = @_;

	my ($out, $pass, $status) = execute("cd $dir; ./problem1");
	if (!$pass) {
		print "Problem 1 FAILED: problem1 execution failed (status $status)";
	} else {
		$$grade++;
		my @res = clean_output($out);
		if (@res != 2) {
			print "Problem 1 FAILED. Expected 2 outputs, got '@res' +1";
		} else {
			$$grade++;
			if (list_any {!looks_like_number $_;} @res) {
				print "Problem 1 FAILED. Expected 2 numeric results, got '@res' +2";
			} else {
				$$grade++;
				if ($res[0] > 0.1) {
					print "Problem 1 FAILED. 2-norm error ($res[0]) > 10% +3";
				} else {
					if ($res[1] != $problem1_solution) {
						print "Problem 1 FAILED. Got '$res[1]', expected '$problem1_solution' +4";
					} else {
						$$grade++;
						print "Problem 1 PASSED +5";
					}
				}
			}
		}
	}
}

sub problem2($$$) {
	my ($dir, $u, $grade) = @_;

	my ($out, $pass, $status) = execute("cd $dir; ./problem2 $u");
	if (!$pass) {
		print "Problem 2 with u=$u FAILED: problem2 execution failed (status $status)";
	} else {
		$$grade++;
		my @res = clean_output($out);
		if (@res != 4) {
			print "Problem 2 with u=$u FAILED. Expected 4 outputs, got '@res' +1";
		} else {
			$$grade++;
			if (list_any {!looks_like_number $_;} @res) {
				print "Problem 2 with u=$u FAILED. Expected 4 numeric results, got '@res' +2";
			} else {
				$$grade++;
				if ($res[0] > 0.1) {
					print "Problem 2 with u=$u FAILED. Frobenius-norm error ($res[0]) > 10% +3";
				} else {
					$$grade++;
					print "Problem 2 with u=$u PASSED +4";
				}
			}
		}
	}
}

my $grade = 0;
# Is there some kind of build system?
if (!detect_build_system($dir)) {
	print "No build system found";
}

# Problem 1	
if (!try_build($dir, "problem1")) {
	print "problem1 not found";
} else {
	print "problem1 built successfully +1";
	$grade++;
	problem1($dir, \$grade);
}

# Problem 2
if (!try_build($dir, "problem2")) {
	print "problem2 not found";
} else {
	print "problem2 built successfully +1";
	$grade++;
	problem2($dir, 0, \$grade);
	problem2($dir, 1, \$grade);
}

my $points = 14;
my $score = int(0.5 + 100.0 * $grade/$points);
printf("grade = %d/%d (%d%%)\n", $grade, $points, $score);
