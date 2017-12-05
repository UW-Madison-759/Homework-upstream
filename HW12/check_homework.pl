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

sub list_equals($$) {
	my ($l1, $l2) = @_;
	return 0 unless @$l1 == @$l2; # same length
	for my $i (0..@$l1-1) {
		return 0 unless $l1->[$i] == $l2->[$i];
	}
	return 1;
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

sub check_pdf($$$$$) {
	my ($dir, $file, $suffix, $notes, $grade) = @_;
	
	if (!-f "$dir/$file") {
		push @{$notes}, "Problem $suffix FAILED: $file not found";
	} else {
		$$grade++;
		push @{$notes}, "found $file, Problem $suffix PASSED +1";
	}	
}

sub doit($$$$$$$$;$) {
	my ($dir, $suffix, $args, $num_outputs, $ans, $notes, $grade, $ans_idx) = @_;
	$ans_idx //= 0;

	my ($out, $pass, $status) = execute("cd $dir; mpiexec -np 2 ./problem$suffix $args");
	if (!$pass) {
		push @{$notes}, "Problem $suffix with args='$args' FAILED: problem$suffix execution failed (status $status)";
	} else {
		$$grade++;
		my @res = clean_output($out);
		if (@res != $num_outputs) {
			my $output = "@res";
			if(length($output) > 50) {
				$output = substr($output, 0, 50) . '...';
			}
			push @{$notes}, "Problem $suffix  with args='$args' FAILED. Expected $num_outputs outputs, got '$output' +1";
		} else {
			$$grade++;
			if (list_any {!looks_like_number $_;} @res) {
				push @{$notes}, "Problem $suffix  with args='$args' FAILED. Expected $num_outputs numeric result, got '@res' +2";
			} else {
				$$grade++;
				my $is_err = sub(){
					my $r = abs($_[0] - $res[$ans_idx]) / $_[0];
					return ($r > 0.1) ? 1 : 0;
				};
				if ($is_err->($ans->[0]) && $is_err->($ans->[1])) {
					local $" = ',';
					push @{$notes}, "Problem $suffix  with args='$args' FAILED. error($res[$ans_idx], one_of(@{$ans})) > 10% +3";
				} else {
					$$grade++;
					push @{$notes}, "Problem $suffix  with args='$args' PASSED +4";
				}
			}
		}
	}
}

my $dir = cwd();
my $grade = 0;
my @notes;

# Is there some kind of build system?
if (!detect_build_system($dir)) {
	print "No build system found" and exit;
}

# Problem 1
if (!try_build($dir, "problem1")) {
	push @notes, "problem1 not found";
} else {
	push @notes, "problem1 built successfully +1";
	$grade++;
	my @sln1 = (1380.1, 5221.02, 10378.26);
	my @sln2 = (101, 101, 138.25);
	doit($dir, "1",  '128', 3, [$sln1[0], $sln2[0]], \@notes, \$grade, 2);
	doit($dir, "1",  '512', 3, [$sln1[1], $sln2[1]], \@notes, \$grade, 2);
	doit($dir, "1", '1024', 3, [$sln1[2], $sln2[2]], \@notes, \$grade, 2);
}
check_pdf($dir, 'problem1.pdf', '1', \@notes, \$grade);

# Problem 2
if (!try_build($dir, "problem2")) {
	push @notes, "problem2 not found";
} else {
	push @notes, "problem2 built successfully +1";
	$grade++;
	my @sln1 = (1380.1, 5221.02, 7486.26);
	my @sln2 = (101, 101, 138.25);
	doit($dir, "2", '128', 3, [$sln1[0], $sln2[0]], \@notes, \$grade, 2);
	doit($dir, "2", '512', 3, [$sln1[1], $sln2[1]], \@notes, \$grade, 2);
	doit($dir, "2", '727', 3, [$sln1[2], $sln2[2]], \@notes, \$grade, 2);
}
check_pdf($dir, 'problem2.pdf', '2', \@notes, \$grade);

my $points = 28;
my $score = int(0.5 + 100.0 * $grade/$points);
push @notes, sprintf("grade = %d/%d (%d%%)", $grade, $points, $score);
print join("\n", @notes), "\n";

