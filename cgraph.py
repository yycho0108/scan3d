import numpy as np
import inspect
import operator
import ast

from cho_util.math import rotation
import time
from profilehooks import profile

class ComputeGraph(object):
    def __init__(self, funcs, cache={}):
        self.funcs_ = self._build(funcs)
        self.cache_ = cache

    def _build(self, funcs):
        out = {}
        for k, v in funcs.items():
            if isinstance(v, str):
                # style : function string
                root = ast.parse(v)
                args = sorted({node.id for node in ast.walk(
                    root) if isinstance(node, ast.Name)})

                # strip globals (such as module names like np=numpy)
                args = [a for a in args if a not in globals()]
                f = eval('lambda {} : {}'.format(','.join(args), v))

                # reformat definition
                out[k] = (f, args)

                # done
                continue

            try:
                # style : (function, explicit_argument)
                f, a = v
                out[k] = (f, a)
            except Exception as e:
                # style : function
                f = v
                a = inspect.getfullargspec(f).args
                out[k] = (f, a)

        return out

    def __contains__(self, k):
        if k in self.cache_:
            return True
        if k in self.funcs_:
            return True
        return False

    def __setitem__(self, k, v):
        self.cache_[k] = v

    def __getitem__(self, k):
        cache = self.cache_
        if k not in cache:
            # build cache
            if k not in self.funcs_:
                msg = 'Attempting to access an invalid compute path : {}'.format(
                    k)
                raise ValueError(msg)
            f, args = self.funcs_[k]
            params = [self.__getitem__(a) for a in args]
            cache[k] = f(*params)
        return cache[k]

    @profile
    def eval(self, k, cache):
        tmp = self.cache_
        self.cache_ = cache
        out =  self.__getitem__(k)
        self.cache_ = tmp
        return out

    def show(self):
        from graphviz import Digraph
        from collections import defaultdict
        dot = Digraph(comment='Compute Graph')

        # build tree
        is_root = {}
        tree = defaultdict(lambda: [])
        for k, v in self.funcs_.items():
            f, a = v

            for a_ in a:
                if k not in tree[a_]:
                    tree[a_].append(k)

            is_root[k] = False
            for a_ in a:
                if a_ not in is_root:
                    is_root[a_] = True

        root = [k for (k,v) in is_root.items() if v]

        # build node height
        node_height = defaultdict(lambda: 0)
        nodes = set(root)
        h = 0
        while nodes:
            for n in nodes:
                node_height[n] = max(node_height[n], h)
            next_nodes = []
            for n in nodes:
                next_nodes.extend(tree[n])
            next_nodes = list(set(next_nodes))
            nodes = next_nodes
            h += 1
        node_height = dict(node_height)

        # invert
        height_node = defaultdict(lambda:[])
        for k,v in node_height.items():
            height_node[v].append(k)
        height_node = dict(height_node)

        # add node layers
        for k in sorted(height_node.keys()):
            with dot.subgraph() as s:
                s.attr(rank='same')
                [s.node( n ) for n in height_node[k]]

        # add edges
        for k, v in self.funcs_.items():
            f, a = v
            [dot.edge(a_, k) for a_ in a]

        # fianlly, visualize
        dot.render('/tmp/cgraph.gv', view=True)

def main():
    funcs = dict(
        x='axis[..., 0]',
        y='axis[..., 1]',
        z='axis[..., 2]',
        c=(np.cos, ['angle']),
        s=(np.sin, ['angle']),
        yz=(operator.mul, ('y', 'z')),
        zx=(operator.mul, ('x', 'z')),
        xy=(operator.mul, ('x', 'y')),
        xx=(np.square, ['x']),
        yy=(np.square, ['y']),
        zz=(np.square, ['z']),
        x11=(np.square, ['yy']),
        xxc=(operator.mul, ('xx', 'c')),
        yyc=(operator.mul, ('c', 'yy')),
        x9=lambda yy, yyc: -yy+yyc+1,
        x13='np.square(zz)',
        x15=lambda yy: 2.0*yy,
        x17=lambda zz: 2.0*zz,
        xxyy=(operator.mul, ('xx', 'yy')),
        cc=(np.square, ['c']),
        x21=(operator.mul, ('x15', 'zz')),
        x22=lambda c: 2.0*c,
        x23=lambda x, yz, s: x*yz*s,
        cy='np.sqrt(-x11*x22 + x11*cc + x11 - x13*x22 + x13*cc + x13 - x15*xxc - x15 + zz*np.square(s) - 4.0*zz*yyc + x17*c - x17 + xxyy*cc + xxyy + cc*x21 + x21 - x22*x23 + 2.0*x23 + 2.0*yyc + 1.0)',
        out0='np.arctan2(x*s - yz*c + yz, -xx + xxc + x9)',
        out1='np.arctan2(s*y + zx*c - zx, cy)',
        out2='np.arctan2(s*z - xy*c + xy, zz*c - zz + x9)',
        out='np.stack([out0,out1,out2], axis=-1)'
    )

    rot = np.random.normal(size=(1000,3))
    angle = np.linalg.norm(rot, axis=-1)
    axis = rot / angle[..., None]
    cache = dict(axis=axis, angle=angle)
    axa = np.concatenate([axis, angle[...,None]], axis=-1)

    g = ComputeGraph(funcs, cache)

    # visualization
    g.show()

    #print(g['out']) # stateful
    ts = []
    ts.append( time.time() )
    for _ in range(1000):
        (g.eval('out', dict(axis=axis, angle=angle))) # stateless
    ts.append( time.time() )
    for _ in range(1000):
        (rotation.euler.from_axis_angle(rot))
    res1 = (g.eval('out', dict(axis=axis, angle=angle)))
    res2 = rotation.euler.from_axis_angle(axa)
    print (res1[0], res2[0])
    print ( np.square(res1 - res2).mean() )
    ts.append( time.time() )
    print ('cgraph vs raw')
    print ( np.diff(ts) )

if __name__ == '__main__':
    main()
