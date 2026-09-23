#ifndef AEESTDERR_H
#define AEESTDERR_H
/* The SDK's values on a non-Hexagon host (AEE_EOFFSET 0). The DSP adds
   0x80000400; htp_graph_desc.h mirrors that split. */
#define AEE_SUCCESS 0
#define AEE_EFAILED 1
#define AEE_ENOMEMORY 2
#define AEE_ECLASSNOTSUPPORT 3
#define AEE_EBADSTATE 13
#define AEE_EBADPARM 14
#define AEE_EBADITEM 16
#define AEE_EINVALIDFORMAT 17
#define AEE_EINCOMPLETEITEM 18
#define AEE_EUNSUPPORTED 20
#define AEE_ENOTYPE 34
#define AEE_EINVALIDITEM 42
#define AEE_EINVHANDLE 44
typedef int AEEResult;
#endif
