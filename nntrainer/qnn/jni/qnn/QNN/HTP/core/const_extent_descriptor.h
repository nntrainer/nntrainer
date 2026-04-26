//==============================================================================
//
// Copyright (c) 2023 Qualcomm Technologies, Inc.
// All Rights Reserved.
// Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#ifndef CONST_EXTENT_DESCRIPTOR_H
#define CONST_EXTENT_DESCRIPTOR_H 1

#include <cstdio>
#include <vector>
#include <cassert>
#include <string>
#include "forward_classes.h"
#include "serialize_defs.h"
#include "pickle_header_tags.h"
#include "const_extent_shared.h"

namespace hnnx {

// This class is used, on both encoder and decoder, to contain a 'const extent descriptor' in its raw form, (just an array of uint32)
// and provide higher-level access to the contents.

class ConstExtentDesc {
  protected:
    using table_t = std::vector<uint32_t>;
    // The 'table' may or may not contain the 'padding' section at the end; this is not accessed,
    // and the serialize method will always generate the required padding.
    table_t table;
    // some values broken out from the header...
    unsigned extab_n = 0, extab_idx = 0; // number of extents, and word index where they start
    unsigned mptab_n = 0, mptab_idx = 0; // number of memory pools, and word index where they start.
    unsigned desc_len = 0; // length of the entire descriptor in bytes (0 if invalid descriptor)

    bool scan_table(); // sanity check, and unpacks the above; returns true if OK.

  public:
    static uint8_t constexpr EXTENT_FLAGS_BITFIELD_LSB = 8;
    static uint8_t constexpr EXTENT_FLAGS_BITFIELD_WIDTH = 8;

    ///
    /// @brief Values for 8b flags in extent record
    ///
    static uint8_t constexpr EXTENT_FLAG_RESERVED_0 = (1 << 0);
    static uint8_t constexpr EXTENT_FLAG_RESERVED_1 = (1 << 1);
    static uint8_t constexpr EXTENT_FLAG_RESERVED_2 = (1 << 2);
    static uint8_t constexpr EXTENT_FLAG_RESERVED_3 = (1 << 3);
    static uint8_t constexpr EXTENT_FLAG_IS_FAR_HINT = (1 << 4); ///< Contents maybe far
    static uint8_t constexpr EXTENT_FLAG_RESERVED_5 = (1 << 5);
    static uint8_t constexpr EXTENT_FLAG_RESERVED_6 = (1 << 6);
    static uint8_t constexpr EXTENT_FLAG_RESERVED_7 = (1 << 7);

    // Return from 'extent_info'.
    struct extab_entry {
        uint32_t extent_flags;
        uint32_t align; // a power of 2, >= 64
        uint64_t offset; // offset, in bytes, from the start of the descriptor, to where the data is.
        uint64_t length; // length of the data in bytes.
    };
    // Return from 'mempool_info'.
    // Note: if 'adjust_offset' is true, the 'offset' field from the containing extent will be added to offset,
    // so that the offset is from the start of the descriptor, instead of the start of the containing extent.
    struct mempool_entry {
        uint32_t mempool_id; // a mempool id >=2 indicating a const mempool
        uint32_t extent_id; // an extent_id, >=1
        uint64_t offset; // offset in bytes of the data from the start of the extent (see note above)
        uint64_t length; // length in bytes of the data
    };
    // optional name of the const_extent this descriptor corresponds to. Used for matching in weight_sharing.
    std::string name = std::string{};

    ConstExtentDesc() {}
    ConstExtentDesc(table_t &&table_in);
    void serialize(Serializer &) const;
    inline bool load_table(table_t &&table_in)
    {
        table = std::move(table_in);
        return scan_table();
    }

    constexpr bool is_valid() const { return desc_len != 0; }

    constexpr unsigned descriptor_length() const { return desc_len; }

    constexpr unsigned num_extents() const { return extab_n; }
    constexpr unsigned num_mempools() const { return mptab_n; }

    // unpack a row of the extent table
    // NOTE: extent_id is 1-based, must be 1 .. num_extents()
    extab_entry extent_info(unsigned extent_id) const;

    // unpack a row of the mempool table.
    // note: idx is not a mempool idx, it is a 1-based row in range 1...num_mempools();
    // if adjust_offset, the offset of the containing extent is added to the offset
    // of the mempool in the returned value.
    mempool_entry mempool_info(unsigned idx, bool adjust_offset = false) const;

    // The ordering of the data and the descriptors is such that:
    //
    // (1)  extent_info(1).offset >= descriptor_length()
    //      mempool_info(1,true).offset >= descriptor_length()
    // (2) for i >=2,
    //      extent_info(i).offset >= extent_info(i+1).offset + extent_info(i+1).length
    //      mempool_info(i,true).offset >= mempool_info(1-1,true).offset + mempool_info(1-1).length
    //

#if !defined(PREPARE_DISABLED)
    ///
    /// @brief Memory pool record iterator
    /// @details Use to iterator over records in memory pool table in constant
    /// extent descriptor
    ///
    class mempool_iterator {
      public:
        using iterator_category = std::input_iterator_tag;
        using value_type = ConstExtentDesc::mempool_entry;
        using difference_type = std::ptrdiff_t;
        using pointer = value_type *;
        using reference = value_type &;

        ///
        /// @brief Constructor
        /// @param [in] cedesc A valid constant extent descriptor instance
        /// @param [in] index Record index (zero-based!)
        ///
        explicit mempool_iterator(ConstExtentDesc const &cedesc, uint32_t index) : _cedesc(cedesc), _index(index) {}

        ///
        /// @brief Increment record
        /// @return Iterator
        ///
        mempool_iterator &operator++()
        {
            // Increment IFF valid constant extent descriptor and mempool record
            // index within range
            _index += (_cedesc.is_valid() && (_index < _cedesc.mptab_n)) ? 1 : 0;
            return *this;
        }

        ///
        /// @brief Equality operator
        /// @return true if iterators are equal
        ///
        bool operator==(mempool_iterator const &other) const { return _index == other._index; }

        ///
        /// @brief Inequality operator
        /// @return true if iterators are not equal
        ///
        bool operator!=(mempool_iterator const &other) const { return !(*this == other); }

        ///
        /// @brief Dereference iterator
        ///
        reference operator*();

      private:
        ///
        /// @brief Reference to a constant extent descriptor instance
        /// @details It contains the blob representing constant extent segment
        ///
        ConstExtentDesc const &_cedesc;

        ///
        /// @brief Current index
        ///
        uint32_t _index;

        ///
        /// @brief Mempool record entry
        /// @details It is assigned when on iterator dereference
        ///
        value_type _entry;
    };

    ///
    /// @brief Return mempool iterator initialized to the first record
    /// @return Mempool iterator
    ///
    mempool_iterator begin() { return mempool_iterator(*this, 0); }

    ///
    /// @brief Return mempool iterator beyond the last record
    /// @warning Intended to be used as a sentinel
    /// @return Mempool iterator
    ///
    mempool_iterator end() { return mempool_iterator(*this, mptab_n); }
#endif
};
#ifndef PREPARE_DISABLED
// Called at the end of serializing a graph, if 'const extent' mode is enabled.
// See comment in const_extent_descriptor.cc for full details.
// LCOV_EXCL_START [SAFTYSWCCB-1542]
size_t write_aligned_const_info(Graph const &gr, Serializer &sctx, unsigned buried_aux_n_words = 0);
#else
inline constexpr size_t write_aligned_const_info(Graph const &gr, Serializer const &sctx, unsigned = 0)
{
    return 0;
}
// LCOV_EXCL_STOP
#endif

} // namespace hnnx

#endif // CONST_EXTENT_DESCRIPTOR_H
