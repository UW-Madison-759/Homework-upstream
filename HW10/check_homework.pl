use strict;
use warnings;
use Cwd qw(cwd);
use Scalar::Util qw(looks_like_number);

sub execute($%) {
	my ($cmd, %args) = @_;
	my $silent = $args{'silent'} ? ' 2>&1 >/dev/null' : '';
	my $ret = `$cmd $silent`;
	return $ret, !(( $? >> 8 ) != 0 || $? == -1 || ( $? & 127 ) != 0), $?;
}
sub try_build($$) {
	my ($dir, $target) = @_;
	unlink "$dir/$target" if -e "$dir/$target";
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
sub check_numeric($$$$$$;$) {
	my ($dir, $suffix, $args, $num_outputs, $ans, $grade, $ans_idx) = @_;
	$ans_idx //= 0;

	my ($out, $pass, $status) = execute("cd $dir; ./problem$suffix $args");
	if (!$pass) {
		print "Problem $suffix with args='$args' FAILED: problem$suffix execution failed (status $status)";
	} else {
		$$grade++;
		my @res = clean_output($out);
		if (@res != $num_outputs) {
			print "Problem $suffix  with args='$args' FAILED. Expected $num_outputs outputs, got '@res' +1";
		} else {
			$$grade++;
			if (list_any {!looks_like_number $_;} @res) {
				print "Problem $suffix  with args='$args' FAILED. Expected $num_outputs numeric result, got '@res' +2";
			} else {
				$$grade++;
				if (abs($ans - $res[$ans_idx]) / $ans > 0.1) {
					print "Problem $suffix  with args='$args' FAILED. error($res[$ans_idx], $ans) > 10% +3";
				} else {
					$$grade++;
					print "Problem $suffix  with args='$args' PASSED +4";
				}
			}
		}
	}
}
sub check_pdf($$$$) {
	my ($dir, $file, $suffix, $grade) = @_;
	
	if (!-f "$dir/$file") {
		print "Problem $suffix FAILED: $file not found";
	} else {
		$$grade++;
		print "found $file, Problem $suffix PASSED +1";
	}	
}

#---------------------------------------------------------------------------------------------------------
my $grade = 0;
my @notes;
my $dir = cwd();
$\ = "\n";

# Is there some kind of build system?
if (!detect_build_system($dir)) {
	print "No build system found";
}

# Problem 1
if (!try_build($dir, "problem1")) {
	push @notes, "problem1 not found";
} else {
	push @notes, "problem1 built successfully +1";
	$grade++;
	my @sln = (637.61, 2568.42, 5143.36);
	check_numeric($dir, "1",  '128', 3, $sln[0], \$grade, 1);
	check_numeric($dir, "1",  '512', 3, $sln[1], \$grade, 1);
	check_numeric($dir, "1", '1024', 3, $sln[2], \$grade, 1);
}
check_pdf($dir, 'problem1.pdf', '1', \$grade);

# Problem 2
if (!try_build($dir, "problem2")) {
	push @notes, "problem2 not found";
} else {
	push @notes, "problem2 built successfully +1";
	$grade++;
	my @sln = (637.61, 2568.42, 5143.36);
	check_numeric($dir, "2",  '128', 3, $sln[0], \$grade, 1);
	check_numeric($dir, "2",  '512', 3, $sln[1], \$grade, 1);
	check_numeric($dir, "2", '1024', 3, $sln[2], \$grade, 1);
}
check_pdf($dir, 'problem2.pdf', '2', \$grade);

# Problem 3
my $extra_credit = 0;
if (try_build($dir, "problem3")) {
	push @notes, "problem3 built successfully +1";
	$extra_credit++;
	my @sln = (637.61, 2568.42, 5143.36);
	check_numeric($dir, "3",  '128', 3, $sln[0], \$extra_credit, 1);
	check_numeric($dir, "3",  '512', 3, $sln[1], \$extra_credit, 1);
	check_numeric($dir, "3", '1024', 3, $sln[2], \$extra_credit, 1);
}
check_pdf($dir, 'problem3.pdf', '1', \$extra_credit);

my $points = 28;
my $score = int(0.5 + 100.0 * $grade/$points) + 50.0 * $extra_credit / 14.0;
print "Extra credit = ($extra_credit/14)";
print sprintf("grade = %d/%d (%d%%)", $grade, $points, $score);
