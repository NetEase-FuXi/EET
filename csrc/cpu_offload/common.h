//
// Created by ligz on 2021/4/30.
//

#ifndef EET_COMMON_H
#define EET_COMMON_H

#define ONE_G  (1024 * 1024 * 1024.0)

namespace eet::co {
    inline int get_itemsize(MetaDesc meta) {
        int itemsize = 0;
        switch (meta.dtype_) {
            case torch::kFloat32:
                itemsize = 4;
                break;
            case torch::kFloat16:
                itemsize = 2;
                break;
                //TODO
            case torch::kInt8:
                itemsize = 1;
                break;
            default:
                break;
        }
        return itemsize;
    }
}
#endif //EET_COMMON_H
