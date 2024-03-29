#!/usr/bin/env python2
import numpy as np
import pdb


class NDRecord(object):
    """
    N-dimensional structured array.
    Implements a dynamic buffer that doubles in size upon overflow.
    """
    C0 = 8

    def __init__(self, dtype, capacity=C0):
        dtype = np.dtype(dtype)
        # assert protected attribute names
        # for kwd in ['dtype_', 'size_', 'cap_', 'data_']:
        #     assert(kwd not in dtype.names)

        self.dtype_ = dtype
        self.size_ = 0
        self.cap_ = capacity
        self.data_ = np.empty(self.cap_, dtype=self.dtype_)

    def __len__(self):
        return self.size_

    def reset(self, delete=True):
        self.size_ = 0
        if delete:
            self.resize(self.cap_)

    def resize(self, new_cap):
        if new_cap < self.cap_:
            self.size_ = min(self.size_, new_cap)
            return
        # handle `expansion` case
        #self.data_ = np.resize(self.data_, self.cap_)
        self.cap_ = int(new_cap)
        data = self.data_
        self.data_ = np.empty(shape=self.cap_, dtype=self.dtype_)
        n_keep = min(self.size_, new_cap)
        self.data_[:n_keep] = data[:n_keep]
        self.size_ = n_keep

    def append(self, rec):
        if self.size_ >= self.cap_:
            self.resize(2*self.cap_)
        self.data_[self.size_] = rec
        self.size_ += 1

    def __getslice__(self, start, stop):
        return self.__getitem__(slice(start, stop))

    def __getitem__(self, *args, **kwargs):
        # pdb.set_trace()
        return self.data.__getitem__(*args, **kwargs)
        # return self.data.__getitem__(*args, **kwargs)

    def __setitem__(self, *args, **kwargs):
        return self.data.__setitem__(*args, **kwargs)

    def extend(self, recs):
        if isinstance(recs, dict):
            # each field specified independently
            if set(recs.keys()) != set(self.dtype_.names):
                raise ValueError("Input format does not match : {} vs. {}".format(
                    recs.keys(), self.dtype_.names
                ))
            n_add = len(recs.values()[0])
            new_size = self.size_+n_add
            if (new_size >= self.cap_):
                self.resize(max(2*self.cap_, new_size))

            d_new = self.data_[self.size_:new_size]
            for k in self.dtype_.names:
                d_new[k] = recs[k]

            self.size_ = new_size
        else:
            # default behavior: treat as list with same axial alignment
            if (self.size_ + len(recs)) >= self.cap_:
                self.resize(max(2*self.cap_, self.size_+len(recs)))
                return self.extend(recs)
            else:
                self.data_[self.size_:self.size_+len(recs)] = recs
                self.size_ += len(recs)

    @property
    def data(self):
        return self.data_[:self.size_]

    @property
    def size(self):
        return self.size_

    @property
    def dtype(self):
        return self.dtype_

    def help(self):
        print(0, self.dtype_)


class DictRecord(object):
    """
    Dictionary-based structured array.
    Implements a dynamic buffer that doubles in size upon overflow.
    Unlike NDRecord, dtype must be specified as a list of (name, type) strings.
    """
    C0 = 8

    def __init__(self, dtype, capacity=C0):
        # metadata
        self.dtype_ = dtype
        self.size_ = 0
        self.cap_ = capacity
        # buffer
        self.data_ = {fmt[0]: None for fmt in self.dtype_}

    def __len__(self):
        return self.size_

    def resize(self, new_cap):
        self.cap_ = new_cap
        for k, v in self.data_.iteritems():
            self.data_[k] = np.resize(v, self.cap_)

    def append(self, rec):
        if self.size_ == self.cap_:
            self.resize(self.cap_*2)
        for fmt in self.dtype_:
            k = fmt[0]
            dtype = fmt[1]
            if self.data_[k] is None:
                entry = np.asarray(rec[k], dtype=dtype)
                self.data_[k] = np.expand_dims(entry, 0)
            else:
                self.data_[k][self.size_] = rec[k]
        self.size_ += 1

    def extend(self, recs):
        for rec in recs:
            self.append(rec)

    @property
    def data(self):
        return {k: self.data_[k][:self.size_] for k in self.data_.iterkeys()}

    @property
    def size(self):
        return self.size_

    @property
    def dtype(self):
        return self.dtype_


def test_dictrec():
    y = DictRecord((
        ('a', np.int32),
        ('b', np.int32)
    ))
    y.append(dict(
        a=15, b=16.0
    ))
    print [v.dtype for v in y.data.values()]
    print y.data

    #y.extend([("xyz", 12, 3.2), ("abc", 100, 0.2)])
    #y.append(("123", 1000, 0))
    #print y.data
    # for i in xrange(100):
    #    y.append((str(i), i, i+0.1))


def test_ndrec():
    y = NDRecord(('a4,int32,float64'))
    y.extend([("xyz", 12, 3.2), ("abc", 100, 0.2)])
    y.append(("123", 1000, 0))
    print y.data
    for i in xrange(100):
        y.append((str(i), i, i+0.1))

    y.extend(dict(
        f0=["def", "ghi", ],
        f1=[153, 556],
        f2=[15.0, 12.3]
    ))

    print '!!'
    print y.data[-2:]
    y.help()

    print y[:2]['f0']
    print y['f0'][:2]

    # slicing works either way
    y[:2]['f0'] = "x"
    print y['f0'][:2]
    y['f0'][:2] = "a"
    print y[:2]['f0']


def main():
    # test_dictrec()
    test_ndrec()


if __name__ == '__main__':
    main()
