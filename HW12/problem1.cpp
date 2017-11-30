#include <iostream>
#include <omp.h>
#define OMPI_SKIP_MPICXX  /* Don't use OpenMPI's C++ bindings (they are deprecated) */
#include <mpi.h>

namespace mpi {
	class context {
		int m_rank, m_size;
	public:
		context(int *argc, char **argv[]) : m_rank { -1 } {
			if (MPI_Init(argc, argv) == MPI_SUCCESS) {
				MPI_Comm_rank(MPI_COMM_WORLD, &m_rank);
				MPI_Comm_size(MPI_COMM_WORLD, &m_size);
			}
		}
		~context() {
			if(m_rank >= 0) {
				MPI_Finalize();
			}
		}
		explicit operator bool() const {
			return m_rank >= 0;
		}
		int rank() const noexcept { return m_rank; }
		int size() const noexcept { return m_size; }
	};
}

int main(int argc, char *argv[]) {
	mpi::context ctx(&argc, &argv);

	if(!ctx) {
		std::cerr << "MPI Initialization failed\n";
		return -1;
	}

	if(ctx.rank() == 0) {
		int x=0;
		constexpr int source_rank = 1;  // We expect a message from Task 1
		MPI_Status status;
		MPI_Recv(&x, 1, MPI_INT, source_rank, 0, MPI_COMM_WORLD, &status);
		std::cout << "Received x = " << x << " on root task.\n";
	} else {
		const int i=4;
		constexpr int dest_rank = 0;  // We send a message to Task 0
		MPI_Send(&i, 1, MPI_INT, dest_rank, 0, MPI_COMM_WORLD);
	}
}
