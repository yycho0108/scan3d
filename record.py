#!/usr/bin/env python2
import numpy as np

class NDRecord(object):
    """
    N-dimensional structured array.
    Implements a dynamic buffer that doubles in size upon overflow.
    """
    C0=8
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

    def resize(self, new_cap):
        self.cap_ = int(new_cap)
        #self.data_ = np.resize(self.data_, self.cap_)
        data = self.data_
        self.data_ = np.empty(shape=self.cap_, dtype=self.dtype_)
        self.data_[:self.size_] = data[:self.size_]

        #for k in self.dtype_.names:
        #    setattr(self, k, self.data_[k])

    def append(self, rec):
        if self.size_ >= self.cap_:
            self.resize(2*self.cap_)
        self.data_[self.size_] = rec
        self.size_ += 1

    def __getitem__(self, *args, **kwargs):
        return self.data.__getitem__(*args, **kwargs)
    def __setitem__(self, *args, **kwargs):
        return self.data.__setitem__(*args, **kwargs)

    def extend(self, recs):
        if (self.size_ + len(recs)) >= self.cap_:
            self.resize(2*self.cap_)
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
    C0=8
    def __init__(self, dtype, capacity=C0):
        # metadata
        self.dtype_ = dtype
        self.size_   = 0
        self.cap_    = capacity
        # buffer
        self.data_ = {fmt[0]:None for fmt in self.dtype_}

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
            k     = fmt[0]
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
        return {k:self.data_[k][:self.size_] for k in self.data_.iterkeys()}

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
    #for i in xrange(100):
    #    y.append((str(i), i, i+0.1))

def test_ndrec():
    y = NDRecord(('a4,int32,float64'))
    y.extend([("xyz", 12, 3.2), ("abc", 100, 0.2)])
    y.append(("123", 1000, 0))
    print y.data
    for i in xrange(100):
        y.append((str(i), i, i+0.1))

    print y.data[-2:]
    y.help()

def main():
    #test_dictrec()
    test_ndrec()

if __name__ == '__main__':
    main()
