"""Conservative static initializer-value equivalence draft. Never executes body.

Only the exact pinned default parent is eligible. `unknown` is not a rejection.
An `equivalent` result concerns returned 78 float64 initializer values on every
scheduled call, conditional on normal NumPy allocation/operation completion.
It does not prove equal runtime, memory, warnings, full-program behavior or cost.
"""
import ast
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

CONTRACT='sciel-circle-baseline-initializer-equivalence/1'
PINS={
 'builder_source_sha256':'4a5345ab00623b2bda418943dc1aaef1ac979edc2775fd165fd1ab84631d8fe7',
 'proposal_builder_sha256':'5def9eb52c65962e62c9269f1f953a715b2ba3adfed791c37a6cf35380c4cbb9',
 'control_source_sha256':'3089f4e36bf787275ade5c3c24d4836ad8dc26e409185c0679c3377f4ce9c5f9',
 'common_scaffold_sha256':'3ac4659923530a3949903d6416c6a19b059623e0095ce0e9f032f7e016498734',
}
ALIASES=('best_params','initial_params','uniform_noise','ordinary_params')
MAX_BODY_BYTES=8192
MAX_NODES=1024
MAX_ARRAY_ALLOCATIONS=256

def encoded(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def digest(x):return hashlib.sha256(x).hexdigest()
LOADED_CHECKER_SHA256=digest(Path(__file__).read_bytes())
class Unknown(Exception):pass

def need(ok,reason):
    if not ok:raise Unknown(reason)

@dataclass(frozen=True)
class Scalar:
    key: str
    low: float
    high: float

@dataclass
class Array:
    storage: list
    indices: tuple
    shape: tuple
    def values(self):return [self.storage[i] for i in self.indices]
    def contiguous(self):return self.indices==tuple(range(self.indices[0],self.indices[0]+len(self.indices))) if self.indices else True

class Machine:
    def __init__(self,domain):
        self.domain=domain;self.allocations=0;self.operations=0
        b=[self.leaf('best',i,-1e-6,1.000001) for i in range(78)]
        initial=b[:] if domain=='zero' else [self.leaf('initial',i,0.,1.) for i in range(78)]
        noise=[self.leaf('noise',i,-.06,.06) for i in range(78)]
        ordinary=[self.add(a,n) for a,n in zip(b,noise)]
        ordinary=[self.clip(v) if i<52 else self.maximum(v) for i,v in enumerate(ordinary)]
        self.original_ordinary=tuple(ordinary)
        self.env=dict(zip(ALIASES,[self.array(b,(78,)),self.array(initial,(78,)),self.array(noise,(78,)),self.array(ordinary,(78,))]))
    def leaf(self,name,index,lo,hi):return Scalar(digest(encoded(['input',name,index])),lo,hi)
    def node(self,op,values,lo,hi):
        need(math.isfinite(lo) and math.isfinite(hi),'potential_nonfinite_intermediate')
        self.operations+=1;need(self.operations<=100000,'symbolic_work_limit')
        return Scalar(digest(encoded([op,*[v.key if isinstance(v,Scalar) else v for v in values]])),lo,hi)
    def add(self,a,b):
        lo=math.nextafter(a.low+b.low,-math.inf);hi=math.nextafter(a.high+b.high,math.inf)
        return self.node('numpy.float64.add',[a,b],lo,hi)
    def maximum(self,x):return self.node('numpy.float64.maximum',[x,float(.001).hex()],max(x.low,.001),max(x.high,.001))
    def clip(self,x):return self.node('numpy.float64.clip',[x,float(.001).hex(),float(.999).hex()],min(max(x.low,.001),.999),min(max(x.high,.001),.999))
    def array(self,values,shape):
        values=list(values);need(math.prod(shape)==len(values) and len(values)<=78,'array_shape_or_capacity')
        self.allocations+=1;need(self.allocations<=MAX_ARRAY_ALLOCATIONS,'symbolic_allocation_limit')
        return Array(values,tuple(range(len(values))),tuple(shape))
    def scalar_constant(self,node):
        need(isinstance(node,ast.Constant) and type(node.value) in (int,float),'numeric_literal_required')
        value=float(node.value);need(math.isfinite(value),'nonfinite_literal');return value
    def dimension(self,node):
        need(isinstance(node,ast.Constant) and type(node.value) is int and node.value>0,'positive_integer_dimension_required')
        return node.value
    def subscript(self,node):
        a=self.expression(node.value);need(isinstance(a,Array) and len(a.shape)==1,'only_one_dimensional_basic_subscript')
        s=node.slice
        if isinstance(s,ast.Constant):
            need(type(s.value) is int and 0<=s.value<len(a.indices),'out_of_range_or_noninteger_index')
            return a.storage[a.indices[s.value]]
        need(isinstance(s,ast.Slice) and s.step is None,'only_unit_step_basic_slice')
        def bound(x,default):
            if x is None:return default
            need(isinstance(x,ast.Constant) and type(x.value) is int and 0<=x.value<=len(a.indices),'slice_bound_unknown')
            return x.value
        start,stop=bound(s.lower,0),bound(s.upper,len(a.indices))
        need(start<=stop,'reversed_slice_unknown')
        selected=a.indices[start:stop]
        return Array(a.storage,selected,(len(selected),))
    def reshape(self,array,args):
        need(isinstance(array,Array) and array.contiguous(),'reshape_requires_known_C_contiguous_array')
        nodes=args[0].elts if len(args)==1 and isinstance(args[0],(ast.Tuple,ast.List)) else args
        need(1<=len(nodes)<=2,'reshape_rank_unknown')
        shape=tuple(self.dimension(n) for n in nodes)
        need(math.prod(shape)==len(array.indices),'reshape_size_mismatch')
        return Array(array.storage,array.indices,shape)
    def expression(self,node):
        if isinstance(node,ast.Name):
            need(node.id in self.env,'unbound_or_forbidden_name');return self.env[node.id]
        if isinstance(node,ast.Constant) and node.value is None:return None
        if isinstance(node,ast.Subscript):return self.subscript(node)
        if isinstance(node,ast.BinOp):
            need(isinstance(node.op,ast.Add),'unsupported_arithmetic')
            left,right=self.expression(node.left),self.expression(node.right)
            need(isinstance(left,Array) and isinstance(right,Array) and left.shape==right.shape,'addition_requires_equal_float64_array_shapes')
            return self.array([self.add(a,b) for a,b in zip(left.values(),right.values())],left.shape)
        need(isinstance(node,ast.Call) and not node.keywords,'unsupported_expression_or_call_keywords')
        f=node.func;need(isinstance(f,ast.Attribute),'dynamic_or_builtin_call_unknown')
        if isinstance(f.value,ast.Name) and f.value.id=='np':
            name=f.attr
            if name=='copy':
                need(len(node.args)==1,'copy_signature');a=self.expression(node.args[0]);need(isinstance(a,Array),'copy_requires_array')
                return self.array(a.values(),a.shape)
            if name in ('maximum','clip'):
                need(len(node.args)==(2 if name=='maximum' else 3),'clamp_signature')
                a=self.expression(node.args[0]);need(isinstance(a,Array),'clamp_requires_array')
                bounds=[self.scalar_constant(n) for n in node.args[1:]]
                need(bounds==([.001] if name=='maximum' else [.001,.999]),'only_original_clamp_bounds')
                return self.array([self.maximum(x) if name=='maximum' else self.clip(x) for x in a.values()],a.shape)
            if name=='reshape':
                need(len(node.args)==2,'np_reshape_signature');return self.reshape(self.expression(node.args[0]),node.args[1:])
            if name=='ravel':
                need(len(node.args)==1,'np_ravel_signature');a=self.expression(node.args[0])
                need(isinstance(a,Array) and a.contiguous(),'ravel_requires_known_C_contiguous_array')
                return Array(a.storage,a.indices,(len(a.indices),))
            raise Unknown('unsupported_numpy_call')
        a=self.expression(f.value);need(isinstance(a,Array),'method_requires_known_array')
        if f.attr=='copy':
            need(not node.args,'array_copy_signature');return self.array(a.values(),a.shape)
        if f.attr=='reshape':return self.reshape(a,node.args)
        if f.attr=='ravel':
            need(not node.args and a.contiguous(),'array_ravel_signature_or_layout')
            return Array(a.storage,a.indices,(len(a.indices),))
        raise Unknown('unsupported_array_method')
    def assign(self,target,value):
        if isinstance(target,ast.Name):
            need(target.id not in ('np','math','restart_index','incumbent','initial','noise','default_guess') and not target.id.startswith('_'),'forbidden_assignment_target')
            self.env[target.id]=value;return
        need(isinstance(target,ast.Subscript),'unsupported_assignment_target')
        # Evaluate target/view only after RHS, as Python does. Snapshot all RHS
        # entries before any write, preserving supported NumPy overlap semantics.
        dest=self.subscript(target)
        need(isinstance(dest,Array) and isinstance(value,Array) and dest.shape==value.shape,'assignment_requires_equal_array_shapes')
        values=value.values()
        for index,v in zip(dest.indices,values):dest.storage[index]=v
    def condition(self,node):
        need(isinstance(node,ast.Compare) and len(node.ops)==len(node.comparators)==1
             and isinstance(node.left,ast.Name) and node.left.id=='restart_index'
             and isinstance(node.comparators[0],ast.Constant) and type(node.comparators[0].value) is int
             and node.comparators[0].value==0 and isinstance(node.ops[0],(ast.Eq,ast.NotEq)),'only_restart_zero_test')
        zero=self.domain=='zero';return zero if isinstance(node.ops[0],ast.Eq) else not zero
    def block(self,statements):
        for i,stmt in enumerate(statements):
            if isinstance(stmt,ast.Assign):
                need(len(stmt.targets)==1,'chained_assignment_unknown')
                self.assign(stmt.targets[0],self.expression(stmt.value))
            elif isinstance(stmt,ast.If):
                selected=stmt.body if self.condition(stmt.test) else stmt.orelse
                found,result=self.block(selected)
                if found:
                    need(i==len(statements)-1,'unreachable_suffix_unknown');return True,result
            elif isinstance(stmt,ast.Return):
                need(i==len(statements)-1,'unreachable_suffix_unknown')
                value=self.expression(stmt.value) if stmt.value is not None else None
                if value is None:return True,self.original_ordinary
                need(isinstance(value,Array) and value.shape==(78,),'return_requires_float64_vector78_or_None')
                return True,tuple(value.values())
            elif isinstance(stmt,ast.Pass):pass
            else:raise Unknown('unsupported_statement')
        return False,None

def preflight(tree):
    allowed=(ast.Module,ast.Assign,ast.If,ast.Return,ast.Pass,ast.Name,ast.Load,ast.Store,ast.Constant,
             ast.Call,ast.Attribute,ast.BinOp,ast.Add,ast.Subscript,ast.Slice,ast.Tuple,ast.List,
             ast.Compare,ast.Eq,ast.NotEq)
    nodes=list(ast.walk(tree));need(len(nodes)<=MAX_NODES,'AST_size_limit')
    need(sum(isinstance(n,ast.If) for n in nodes)<=1,'multiple_or_nested_conditionals_unknown')
    for n in nodes:
        need(isinstance(n,allowed),'unsupported_syntax')
        if isinstance(n,ast.Call):
            need(not n.keywords and isinstance(n.func,ast.Attribute),'unsupported_call_form')
            f=n.func
            if isinstance(f.value,ast.Name) and f.value.id=='np':need(f.attr in ('copy','reshape','ravel','maximum','clip'),'unsupported_numpy_call')
            else:need(f.attr in ('copy','reshape','ravel'),'unsupported_array_method')
        if isinstance(n,ast.Attribute):need(not n.attr.startswith('_'),'private_attribute_unknown')
        if isinstance(n,ast.Name):need(not n.id.startswith('_'),'private_name_unknown')

def verify(body,parent_source,*,builder_source,proposal_source):
    """Return finite JSON. No non-equivalence/rejection verdict is produced."""
    result={'schema':CONTRACT,'verdict':'unknown','reason':None,'proof':None}
    try:
        need(digest(Path(__file__).read_bytes())==LOADED_CHECKER_SHA256,'loaded_checker_source_drift')
        need(type(body) is str and len(body.encode())<=MAX_BODY_BYTES and len(body.splitlines())<=512,'body_size_or_type')
        need(type(parent_source) is str,'parent_source_type')
        need(type(builder_source) is bytes and type(proposal_source) is bytes,'trusted_source_bytes_required')
        need(digest(parent_source.encode())==PINS['control_source_sha256'],'parent_is_not_exact_default_control')
        need(digest(builder_source)==PINS['builder_source_sha256'] and digest(proposal_source)==PINS['proposal_builder_sha256'],'trusted_builder_or_scaffold_identity_unknown')
        tree=ast.parse(body,mode='exec');preflight(tree)
        states=[]
        for domain in ('zero','positive_multiple_of_four'):
            machine=Machine(domain);returned,value=machine.block(tree.body)
            # Falling off the end is Python None, but explicit return is required
            # by this recognizer to keep its admitted language narrow.
            need(returned,'explicit_return_required')
            matches=[a.key==b.key for a,b in zip(value,machine.original_ordinary)]
            need(len(matches)==78 and all(matches),'initializer_equality_not_proven')
            states.append({'domain':'restart_index == 0; initial values equal best values, separate storage'
                if domain=='zero' else 'restart_index in {4,8,12,...}; initial and best are independent arrays',
                'elements_proven_equal':78,'output_expression_sha256':digest(encoded([x.key for x in value])),
                'operations':machine.operations,'bounded_array_allocations':machine.allocations})
        proof={'schema':'sciel-initializer-value-equivalence-proof/1','checker_contract':CONTRACT,
            'identity':dict(PINS),'parent_source_sha256':digest(parent_source.encode()),'body_sha256':digest(body.encode()),
            'states':states,'scope':'Initializer values only, conditional on successful standard NumPy float64 operations and allocations. Not runtime, resource, timing, warnings, full-program or quality equivalence.',
            'assumptions':['Caller has already accepted this exact raw body using the pinned strict builder/codec; this checker is not an admissibility or executability validator.',
                'Pinned scaffold supplies writable detached C-contiguous float64 arrays of shape78.',
                'Original ordinary vector is precomputed before body: best+same noise, radii maximum .001, centers clip .001/.999.',
                'None returns a copy of the saved outside-body ordinary vector even after local input mutations.',
                'Only calls with restart_index divisible by4 occur; at index0 best and initial have equal values.',
                'Incumbent coordinates/radii are finite and bounded by pinned feasibility checks; uniform noise lies within [-.06,.06].'],
            'checker_source_sha256':LOADED_CHECKER_SHA256}
        proof['sha256']=digest(encoded(proof));result.update(verdict='equivalent',reason='recognized_dataflow_matches_original_initializer',proof=proof)
    except (Unknown,SyntaxError,ValueError,TypeError,OverflowError,RecursionError,OSError) as exc:
        result['reason']=str(exc) if isinstance(exc,Unknown) else type(exc).__name__
    return result
