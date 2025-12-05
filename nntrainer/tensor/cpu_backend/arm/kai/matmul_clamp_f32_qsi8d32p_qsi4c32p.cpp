 const size_t num_blocks = k / bl;
    const size_t num_bytes_per_block_qs4c32 = (bl / 2) + sizeof(int16_t);
    const size_t num_bytes_per_block_qs8c32 = bl + sizeof(int16_t);

    const size_t lhs_native_size_f32 = m * k * sizeof(float);
    const size_t rhs_native_size_f32 = n * k * sizeof(float);
    const size_t rhs_native_size_qs4c32 = n * num_blocks * num_bytes_per_block_qs4c32;

    quant_qs4c32_f32(n, k, bl, (const float*)rhs_native_mtx_f32, (uint8_t*)rhs_native_mtx_qs4c32);

    //----------- MICRO-KERNELS TESTS
    //------------------------------------
    //------------------------------------
    for (size_t idx_variant = 0; idx_variant < num_ukernel_variants; ++idx_variant) {
        std::cout << "Testing " << ukernel_variants[idx_variant].name << std::endl;

        // Get the packing parameters
        const size_t mr = ukernel_variants[idx_variant].ukernel.get_mr();
        const size_t nr = ukernel_variants[idx_variant].ukernel.get_nr();
        const size_t kr = ukernel_variants[idx_variant].ukernel.get_kr();
        const size_t sr = ukernel_variants[idx_variant].ukernel.get_sr();

        // Get the size in bytes for the packed matrices
        const size_t lhs_packed_size = kai_get_lhs_packed_size_lhs_quant_pack_qsi8d32p_f32(m, k, bl, mr, kr, sr);
        const size_t rhs_packed_size =
            kai_get_rhs_packed_size_rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0(n, k, nr, kr, bl);
        const size_t dst_size = ukernel_variants[idx_variant].ukernel.get_dst_size(m, n);

        // Allocate the matrices
        uint8_t* lhs_packed_mtx_qs8d32 = new uint8_t[lhs_packed_size];
        uint8_t* rhs_packed_mtx_qs4c32 = new uint8_t[rhs_packed_size];
        uint8_t* dst_act_mtx_f32 = new uint8_t[dst_size];

        // If the RHS matrix contains constant values, the packing can be performed
        // only once
        struct kai_rhs_pack_qs4cxs1s0_param params;
        params.lhs_zero_point = 1;
        params.rhs_zero_point = 8;

        // RHS packing. RHS is usually constant and packed only once, e.g. at model loading time.
        kai_run_rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0(
            1, n, k,                                  // Dimensions
            nr, kr, sr,                               // Packing arguments
            bl,                                       // Block length
            (const uint8_t*)(rhs_native_mtx_qs4c32),  // RHS
            NULL,                                     // Bias
            rhs_packed_mtx_qs4c32,                    // RHS packed
            0, &params);

        // Worker thread function. Multithreads both LHS packing and matmul execution.
        auto thread_worker = [&](int thread_index) {
            // Each thread processes m_to_process number of rows. Note that, due to the format block size, each
            // thread must process an integer multiple of m_step number of rows.
            const size_t m_step = ukernel_variants[idx_variant].ukernel.get_m_step();
            const size_t num_m_per_thread = kai_roundup(m, m_step * num_threads) / num_threads;
            const size_t m_start = thread_index * num_m_per_thread;

            // For small shapes and m_step > 1, there may not be enough parallelism to put all threads to work
            if (m_start < m) {
                size_t m_to_process = num_m_per_thread;
                if (m_start + m_to_process > m) {
                    m_to_process = m - m_start;
                }

                // LHS packing
                const float* src_ptr = (float*)lhs_native_mtx_f32 +
                    kai_get_lhs_offset_lhs_quant_pack_qsi8d32p_f32(m_start, k * sizeof(float)) / sizeof(float);
                const size_t lhs_packed_offset =
                    ukernel_variants[idx_variant].ukernel.get_lhs_packed_offset(m_start, k, bl);
                void* lhs_packed_ptr = lhs_packed_mtx_qs8d32 + lhs_packed_offset;

                kai_run_lhs_quant_pack_qsi8d32p_f32(
                    m_to_process, k, bl,  // Dimensions
                    mr, kr, sr, 0,        // Packing arguments
                    src_ptr,              // LHS
                    k * sizeof(float),    // LHS stride
                    lhs_packed_ptr);      // LHS packed

                // Matmul micro-kernel
                const size_t dst_stride = n * sizeof(float);
                const size_t rhs_packed_offset = ukernel_variants[idx_variant].ukernel.get_rhs_packed_offset(0, k, bl);
                const void* rhs_packed_ptr = rhs_packed_mtx_qs4c32 + rhs_packed_offset;
                const size_t dst_offset = ukernel_variants[idx_variant].ukernel.get_dst_offset(m_start, 0, dst_stride);
                float* dst_ptr = (float*)(dst_act_mtx_f32 + dst_offset);

                ukernel_variants[idx_variant].ukernel.run_matmul(
                    m_to_process, n, k, bl,  // Dimensions
                    lhs_packed_ptr,          // LHS packed
                    rhs_packed_ptr,          // RHS packed
                    dst_ptr,                 // DST
                    dst_stride,              // DST stride (row)
                    sizeof(float),           // DST stride (col)
                    -FLT_MAX, FLT_MAX        // Min and max for the clamp operation
                );
            }
        };

        const auto time_s = std::chrono::high_resolution_clock::now();

        std::vector<std::thread> threads;
        threads.reserve(num_threads);

        // Create worker threads and execute the operator
        for (int i = 0; i < num_threads; ++i) {
            threads.emplace_back(thread_worker, i);
        }

        // Wait until all threads have finished
        for (auto& t : threads) {
            t.join();
        }
    }