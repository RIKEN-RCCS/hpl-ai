#if defined(__FUJITSU) || defined(__CLANG_FUJITSU)
void remap330(int& row, int& col, int coords[6]); // for 22x20x24
void remap360(int& row, int& col, int coords[6]); // for 24x20x24
void remap392(int& row, int& col, int coords[6]); // for 24x22x24
#endif
