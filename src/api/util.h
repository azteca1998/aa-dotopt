#ifndef DOTOPT__API__UTIL_H
#define DOTOPT__API__UTIL_H


#define _assert(cond, exc, msg) \
    if (!(cond)) \
    { \
        PyErr_SetString((exc), (msg)); \
        return NULL; \
    } \
    do {} while (0)


#endif /* DOTOPT__API__UTIL_H */
