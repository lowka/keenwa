## Keenwa 

Cost-based query optimizer (WIP).

### Features

- [x] Basic algorithm based on Cascades/Columbia papers.
- [x] Basic logical/physical plans.
- [x] Basic implementation/transformation rules.
- [x] Basic physical properties/enforcers.
- [x] Basic cost-model.
- [x] Basic statistics.
- [x] Basic catalog API.

### Physical properties/Enforcers

- [x] Ordering.
- [ ] Partitioning.
- TODO

### TODO

- [ ] Search space pruning (upper bound/lower bound etc).
- [ ] Parameter placeholders/Prepared statements.
- [ ] multi-stage optimization setup (logical rewrites -> cost-based -> .. etc.).
- [ ] generate plans that can be changed at runtime (configurable access-path selection, etc.).
- [ ] Termination: timeout
- [ ] Detect cycles in Optimizer::optimize.
- [ ] Fallback mechanism to build the remaining physical operators when the optimizer fails to build a logical plan.
- ...

---

### Relational operators

- [x] Select/Filter.
- [x] Projection.
- [x] Scan.
- [x] Join.
- [x] Union/Except/Intersect (ALL).
- [x] Limit/Offset
- [ ] Fetch
- [ ] VALUES operator (?).
- [x] Distinct option in SELECT
- TODO

#### Aggregation

- [x] Basic aggregation functions.
- [x] GROUP BY.
- [x] AGGR(..) FILTER (WHERE _).
- [x] AGGR(DISTINCT _).
- [ ] count(*) vs count(col).
- [ ] Window functions.
- [ ] User-defined Aggregate functions.
- [ ] GROUPING SET.

#### Implementation

- [x] HashAggregate
- [ ] Aggregate over sorted data.
- TODO

#### Joins

- [x] Inner
- [x] Left
- [x] Right
- [x] Full
- [x] Cross
- [ ] Semi 
- [ ] Anti (NOT IN ..)
- [ ] Lateral

- [x] Basic transformation rules: AxB -> BxA, (AxB)xC -> Ax(BxC)

#### Implementation

- [x] HashJoin
- [x] MergeSortJoin
- [x] NestedLoopJoin
- TODO

#### Subqueries

- [x] Basic support.
- [x] Correlated subqueries: EXISTS, IN (cases where an input operator is a restriction(filter/select) or a projection).  

### Scalar expressions

- [x] Basic scalar expressions.
- [x] Basic aggregate functions.
- [x] Basic sub queries. 
- [x] Basic identifiers (`table.column`, etc)
- [ ] Compound identifiers (`schema.object.whatever`, etc)
- [x] Functions expressions.
- [x] IS NULL/IS NOT NULL.
- [x] IN/NOT IN (list).
- [x] IN/NOT IN `<subquery>`.
- [x] EXISTS/NOT EXISTS `<subquery>`.
- [ ] ANY/ALL `<subquery>`.
- [x] BETWEEN.
- [x] CASE expression.
- [x] LIKE/NOT LIKE
- [ ] Tuples.
- [ ] Array access.
- [ ] Window functions.
- [ ] User-defined aggregate functions.
- TODO

#### Binary operators

- [x] Basic operators (AND, OR, =, !=, >, <, >=, >=, +, -, /, *, %, ).
- [ ] Bitwise operators.

#### Data types

- [x] Basic data type (string, int32, bool).
- [ ] Floating point types.
- [ ] Decimal types.
- [ ] Byte arrays.
- [x] Date (Days), Time (Hours, Minutes, Seconds, Millis)
- [x] Timestamp with timezone, Timestamp without time zone.
- [x] Time intervals: Year, Year to Month, Month, Day, Day to Hour, Day to Minute, Day to Second.
- [ ] Arrays.
- [ ] Tuples.

#### Functions

- [ ] Additional String functions.
- [ ] Additional numeric functions.
- [ ] Additional date/time functions.
- [ ] Additional interval functions.
- [ ] Additional array functions.

---

### Logical rewrites (Experimental)

- [x] Basic logical rewrites API.
- [x] Predicate push-down.
- [x] Redundant projections removal.
- TODO

---

### Catalog

- [x] Basic tables.
- [x] Basic indexes.
- [ ] Table constraints.
- [ ] More index types.
- [ ] User-defined functions
- TODO

### Statistics

- [x] row count.
- [x] predicate selectivity.
- TODO

---

### Cost model

- [x] basic cost model (row count, predicate selectivity).
- [ ] compute predicate selectivity.
- TODO

---

### Data manipulation operators

TODO

---

### SQL-support (sqlparser-rs)

See test cases in `sql/*_tests.yaml`.

- [x] Basic operators.
- [x] Basic subqueries.
- [x] Joins: Inner, Left, Right, Full, Cross

- [x] SELECT DISTINCT
- [x] LIMIT/OFFSET
- [ ] FETCH
- [ ] EXPLAIN/ANALYZE
- [x] ALIAS (column list)
- [x] AGGREGATE DISTINCT
- [ ] VALUES(..)

- [x] NOT IN/ IN `<subquery>`
- [x] EXISTS/NOT EXISTS `<subquery>`.

#### Basic DML operators

TODO

#### Basic DDL operators

TODO

---