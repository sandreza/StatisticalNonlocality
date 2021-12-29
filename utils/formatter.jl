using JuliaFormatter

kwargs = (;
    indent = 4,
    margin = 80,
    always_for_in = true,
    whitespace_typedefs = true,
    whitespace_ops_in_indices = true,
    remove_extra_newlines = true,
    import_to_using = false,
    pipe_to_function_call = false,
    always_use_return = false, # false plays nicely with KernelAbstractions
    whitespace_in_kwargs = true,
    annotate_untyped_fields_with_any = false,
    short_to_long_function_def = false, # false allows for one-liners
    format_docstrings = false,
    align_assignment = true,
    align_struct_field = true,
    align_conditional = true,
    align_pair_arrow = true,
    align_matrix = true,
    normalize_line_endings = "auto",
    trailing_comma = true,
    join_lines_based_on_source = false, #false forces either single or multiline
    indent_submodule = true,
)

format("."; kwargs...)
