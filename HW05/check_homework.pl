use strict;
use warnings;

use Scalar::Util qw(looks_like_number);
use List::Util qw(any);
use File::Copy qw(copy);
use Cwd qw(cwd);

my @hist_result = qw(307843 309623 308238 309248 308684 307790 308574);
my $dir = cwd();

$\ = "\n";

# Is there some kind of build system?
if (!&detect_build_system($dir)) {
	print "No build system found";
}

if (!&try_build($dir, "problem1")) {
	print "problem1 not found";
} else {
	&problem1(1);
	&problem1(3);
}

if (!-f "$dir/problem1.pdf") {
	print "Problem 1 FAILED: problem1.pdf not found";
} else {
	
	print "found problem1.pdf, Problem 1 PASSED";
}

if (!&try_build($dir, "problem2")) {
	print "problem2 not found";
}

if (!-f "$dir/problem2a.pdf") {
	print "Problem 2 FAILED: problem2a.pdf not found";
} else {
	print "found problem2a.pdf, Problem 2 PASSED";
}

if (!-f "$dir/problem2b.pdf") {
	print "Problem 2B FAILED: problem2b.pdf not found";
} else {
	print "found problem2b.pdf, Problem 2B PASSED";
}

#------------------------------------------------------------------------------------

sub problem1($$) {
	my ($N) = @_;
	
	my ($out, $pass, $status) = &execute("./problem1 $N");
	if (!$pass) {
		print "Problem 1 with N=$N FAILED: problem1 execution failed (status $status)";
	} else {
		my @res = clean_output($out);
		if (@res != 9) {
			print "Problem 1 with N=$N FAILED. Expected 9 outputs, got '@res'";
		} else {
			if (any {!looks_like_number $_;} @res) {
				print "Problem 1 with N=$N FAILED. Expected 9 numeric results, got '@res'";
			} else {
				my @hist = @res[0...6];
				if (!list_equals(\@hist, \@hist_result)) {
					print "Problem 1 with N=$N FAILED. Histogram incorrect. Got '@hist', expected '@hist_result'";
				} else {
					print "Problem 1 with N=$N PASSED";
				}
			}
		}
	}
}

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