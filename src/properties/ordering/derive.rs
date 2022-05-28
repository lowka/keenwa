use crate::meta::ColumnId;
use crate::properties::{OrderingChoice, OrderingColumn};

/// Derives an ordering of columns for input operators given the required ordering.
///
/// Returns the left ordering and the right ordering that should be applied to the input operators
/// in order to satisfy the required ordering. Otherwise returns `None` this means that the required ordering
/// can not be satisfied by sorting the inputs by the given columns.
///
/// This function can be used to implement properties of physical operators that require
/// their inputs to be ordered (such as MergeSortJoin and Unique).
///
/// # Examples
///
/// * `MergeSortJoin` operator for expression JOIN ON a1 = b2 AND a2 = b1 must have
/// `left_columns` = `[a1, a2]` and `right_columns` = `[b2, b1]`.
///
/// * `Unique` operator for input relations A `[a1, a2]` and B `[b1, b2]` must have
/// `left_columns` = `[a1, a2]` and `right_columns` = `[b1, b2]`.
///
pub fn derive_input_orderings(
    ordering: &OrderingChoice,
    left_columns: &[ColumnId],
    right_columns: &[ColumnId],
) -> Option<(OrderingChoice, OrderingChoice)> {
    // Merge Sort Join operator can provide the required ordering if
    // the given ordering satisfies the following properties:
    //  - ordering includes columns used in the join condition
    //  - equivalent columns from the join condition must be used exactly once (2)
    //  - BUT the same column with the same ordering can be - it is simply ignored (3)
    //
    // 1) JOIN A, B ON a1 = b2 AND a3 = b4
    //     ORDER BY a1,b4
    //   Can be implemented via MergeSortJoin.
    //
    // 2) JOIN A, B ON a1 = b2 AND a3 = b2
    //    ORDER BY a1,b2
    //   Can not be implemented via MergeSortJoin without additional orderings
    //   which eliminates all the benefits of a Merge Sort Join.
    //
    // 3) JOIN A, B ON a1 = b1 AN a2 = b2
    //    ORDER BY a1, b2, a1
    //   Can be implemented via MergeSortJoin. The send order by a1 is
    //
    // In the first case the required ordering (a1, a3) is a prefix of
    // the ordering (a1, b4)/(a1, a3) (since a3 is equal to a b4) provided by the merge join operator.
    //
    // And in the second case the ordering (a1, b2) is not a prefix of the ordering
    // provided by the merge sort join.
    //

    let mut columns: Vec<_> = left_columns.iter().copied().zip(right_columns.iter().copied()).collect();

    #[derive(Default)]
    struct OrderingCheckState {
        provided_ordering: Vec<(OrderingColumn, OrderingColumn)>,
        used_columns: Vec<ColumnId>,
        can_not_provide_ordering: bool,
    }

    let mut check_state = OrderingCheckState::default();

    for ord in ordering.columns() {
        if check_state.can_not_provide_ordering {
            break;
        }

        let pos = columns.iter().position(|(l, r)| l == &ord.column() || r == &ord.column());
        if let Some(pos) = pos {
            let (l, r) = columns.remove(pos);
            let pair = if l == ord.column() {
                let r = OrderingColumn::ord(r, ord.descending());
                (*ord, r)
            } else {
                let l = OrderingColumn::ord(l, ord.descending());
                (l, *ord)
            };

            if check_state.used_columns.contains(&l) || check_state.used_columns.contains(&r) {
                // If either left or right column has already been seen in the ordering ->
                // Merge Sort join can not be used
                check_state.can_not_provide_ordering = true;
            } else {
                check_state.used_columns.push(l);
                check_state.used_columns.push(r);
                check_state.provided_ordering.push(pair);
            }
        } else {
            // If such ordering column already exists -> allow duplicate
            // Otherwise the column is not found in join condition -> Merge Sort join
            // can not be used
            if !check_state.provided_ordering.iter().any(|(l_ord, r_ord)| l_ord == ord || r_ord == ord) {
                check_state.can_not_provide_ordering = true
            }
        }
    }

    if check_state.can_not_provide_ordering {
        None
    } else if let Some(ord) = check_state.provided_ordering.last().copied() {
        // Add ordering by the remaining columns using some direction.
        // We chose the sort direction of the last column in the required ordering.
        let (left, right): (Vec<_>, Vec<_>) = check_state
            .provided_ordering
            .into_iter()
            .chain(columns.into_iter().map(|(l, r)| {
                let left_ord = OrderingColumn::ord(l, ord.0.descending());
                let right_ord = OrderingColumn::ord(r, ord.1.descending());
                (left_ord, right_ord)
            }))
            .unzip();

        Some((OrderingChoice::new(left), OrderingChoice::new(right)))
    } else {
        None
    }
}

#[cfg(test)]
mod test {
    use crate::meta::testing::TestMetadata;
    use crate::meta::ColumnId;
    use crate::properties::ordering::derive::derive_input_orderings;
    use crate::properties::{OrderingChoice, OrderingColumn};

    #[test]
    fn single_column_ordering() {
        let mut metadata = TestMetadata::with_tables(vec!["A", "B"]);
        metadata.add_columns("A", vec!["a1", "a2"]);
        metadata.add_columns("B", vec!["b1", "b2"]);

        let test_ordering = TestDeriveOrdering {
            metadata: &metadata,
            ordering: vec!["A:+a1"],
            input: vec![("A:a1", "B:b1")],
            left: "A",
            right: "B",
        };
        test_ordering.expect_ordering(vec!["A:+a1"], vec!["B:+b1"]);
    }

    #[test]
    fn multi_column_ordering() {
        let mut metadata = TestMetadata::with_tables(vec!["A", "B"]);
        metadata.add_columns("A", vec!["a1", "a2"]);
        metadata.add_columns("B", vec!["b1", "b2"]);

        let test_ordering = TestDeriveOrdering {
            metadata: &metadata,
            ordering: vec!["A:+a1", "A:+a2"],
            input: vec![("A:a1", "B:b1"), ("A:a2", "B:b2")],
            left: "A",
            right: "B",
        };
        test_ordering.expect_ordering(vec!["A:+a1", "A:+a2"], vec!["B:+b1", "B:+b2"]);
    }

    #[test]
    fn resulting_ordering_only_depends_on_required_ordering() {
        fn test_ordering(ordering: Vec<&str>) {
            let mut metadata = TestMetadata::with_tables(vec!["A", "B"]);
            metadata.add_columns("A", vec!["a1", "a2"]);
            metadata.add_columns("B", vec!["b1", "b2"]);

            let test_ordering = TestDeriveOrdering {
                metadata: &metadata,
                ordering: vec!["A:+a1", "A:+a2"],
                input: vec![("A:a1", "B:b1"), ("A:a2", "B:b2")],
                left: "A",
                right: "B",
            };
            test_ordering.expect_ordering(vec!["A:+a1", "A:+a2"], vec!["B:+b1", "B:+b2"]);
        }

        test_ordering(vec!["A:+a1", "A:+a2"]);
        test_ordering(vec!["B:+b1", "A:+a2"]);
        test_ordering(vec!["A:+a1", "B:+b2"]);
        test_ordering(vec!["B:+b1", "B:+b2"]);
    }

    #[test]
    fn extend_ordering_by_the_remaining_input_columns() {
        let mut metadata = TestMetadata::with_tables(vec!["A", "B"]);
        metadata.add_columns("A", vec!["a1", "a2"]);
        metadata.add_columns("B", vec!["b1", "b2"]);

        let test_ordering = TestDeriveOrdering {
            metadata: &metadata,
            ordering: vec!["A:+a1"],
            input: vec![("A:a1", "B:b1"), ("A:a2", "B:b2")],
            left: "A",
            right: "B",
        };
        test_ordering.expect_ordering(vec!["A:+a1", "A:+a2"], vec!["B:+b1", "B:+b2"]);
    }

    #[test]
    fn do_not_provide_ordering_when_the_required_ordering_contains_not_sortable_columns() {
        let mut metadata = TestMetadata::with_tables(vec!["A", "B"]);
        metadata.add_columns("A", vec!["a1", "a2"]);
        metadata.add_columns("B", vec!["b1", "b2"]);

        let test_ordering = TestDeriveOrdering {
            metadata: &metadata,
            ordering: vec!["A:+a1", "B:+b2"],
            input: vec![("A:a1", "B:b1")],
            left: "A",
            right: "B",
        };
        test_ordering.expect_do_not_provide_ordering();
    }

    #[test]
    fn do_not_provide_input_ordering_if_equivalent_column_is_present() {
        let mut metadata = TestMetadata::with_tables(vec!["A", "B"]);
        metadata.add_columns("A", vec!["a1", "a2"]);
        metadata.add_columns("B", vec!["b1", "b2"]);

        let test_ordering = TestDeriveOrdering {
            metadata: &metadata,
            ordering: vec!["A:+a1", "B:+b1"],
            input: vec![("A:a1", "B:b2")],
            left: "A",
            right: "B",
        };
        test_ordering.expect_do_not_provide_ordering();
    }

    #[test]
    fn do_not_provide_input_ordering_to_children_when_column_has_already_been_used() {
        let mut metadata = TestMetadata::with_tables(vec!["A", "B"]);
        metadata.add_columns("A", vec!["a1", "a2"]);
        metadata.add_columns("B", vec!["b1", "b2"]);

        //JOIN A, B ON a1 = b2 AND a2 = b2
        //ORDER BY a1, b2
        let test_ordering = TestDeriveOrdering {
            metadata: &metadata,
            ordering: vec!["A:+a1", "B:+b2"],
            input: vec![("A:a1", "B:b2"), ("A:a2", "B:b2")],
            left: "A",
            right: "B",
        };
        test_ordering.expect_do_not_provide_ordering();
    }

    #[test]
    fn propagate_ordering_when_ordering_contains_duplicate_columns() {
        // JOIN A, B ON a1 = b1 AN a2 = b2
        // ORDER BY a1, b2, a1[b2]
        fn test_ordering(ord: Vec<&str>) {
            let mut metadata = TestMetadata::with_tables(vec!["A", "B"]);
            metadata.add_columns("A", vec!["a1", "a2"]);
            metadata.add_columns("B", vec!["b1", "b2"]);

            let test_ordering = TestDeriveOrdering {
                metadata: &metadata,
                ordering: vec!["A:+a1", "B:+b2"].into_iter().chain(ord).collect(),
                input: vec![("A:a1", "B:b1"), ("A:a2", "B:b2")],
                left: "A",
                right: "B",
            };
            test_ordering.expect_ordering(vec!["A:+a1", "A:+a2"], vec!["B:+b1", "B:+b2"]);
        }

        test_ordering(vec!["A:+a1"]);
        test_ordering(vec!["B:+b2"]);
        test_ordering(vec!["A:+a1", "A:+a1"]);
        test_ordering(vec!["A:+a1", "B:+b2"]);
    }

    struct TestDeriveOrdering<'a> {
        metadata: &'a TestMetadata,
        ordering: Vec<&'a str>,
        input: Vec<(&'a str, &'a str)>,
        left: &'a str,
        right: &'a str,
    }

    impl<'a> TestDeriveOrdering<'a> {
        fn prepare_columns(&mut self) -> (Vec<ColumnId>, Vec<ColumnId>) {
            let left_columns = self
                .input
                .iter()
                .map(|(l, _)| {
                    let (table, column) = l.split_once(":").unwrap();
                    self.metadata.find_column(table, column)
                })
                .collect();

            let right_columns = self
                .input
                .iter()
                .map(|(_, r)| {
                    let (table, column) = r.split_once(":").unwrap();
                    self.metadata.find_column(table, column)
                })
                .collect();

            (left_columns, right_columns)
        }

        fn expect_do_not_provide_ordering(mut self) {
            let ordering = build_ordering(self.metadata, &self.ordering);
            let (left, right) = self.prepare_columns();

            let result = derive_input_orderings(&ordering, &left, &right);
            assert_eq!(result, None, "no child ordering");
        }

        fn expect_ordering(mut self, expected_left_ordering: Vec<&str>, expected_right_ordering: Vec<&str>) {
            let ordering = build_ordering(self.metadata, &self.ordering);
            let (left, right) = self.prepare_columns();

            let result = derive_input_orderings(&ordering, &left, &right);
            let left_ordering = build_ordering(self.metadata, &expected_left_ordering);
            let right_ordering = build_ordering(self.metadata, &expected_right_ordering);

            assert_eq!(result, Some((left_ordering, right_ordering)), "ordering");
        }
    }

    fn build_ordering(metadata: &TestMetadata, columns: &[&str]) -> OrderingChoice {
        let columns = columns
            .iter()
            .map(|name| {
                let (table, ord) = name.split_once(":").unwrap();
                let direction = &ord[0..1];
                let descending = match direction {
                    "+" => false,
                    "-" => true,
                    _ => panic!("Unexpected column ordering. Expected +/- but got '{}'", direction),
                };
                let id = metadata.find_column(table, &ord[1..]);
                OrderingColumn::ord(id, descending)
            })
            .collect();
        OrderingChoice::new(columns)
    }
}
