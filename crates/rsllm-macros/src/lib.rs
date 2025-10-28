//! Procedural macros for RSLLM tool calling
//!
//! This crate provides the `#[tool]` and `#[arg]` attribute macros for easy tool definition.

use convert_case::{Case, Casing};
use proc_macro::TokenStream;
use quote::quote;
use syn::{
    parse::{Parse, ParseStream},
    parse_macro_input, ItemFn, ItemStruct, LitStr, Token,
};

/// The `#[arg]` attribute for marking individual tool parameters
///
/// Usage: `#[arg(description = "Parameter description")]`
///
/// This is a marker attribute that the `#[tool]` macro looks for.
/// It doesn't do anything on its own.
#[proc_macro_attribute]
pub fn arg(_args: TokenStream, input: TokenStream) -> TokenStream {
    // This is a pass-through attribute
    // The #[tool] macro will process it
    input
}

/// The `#[context]` attribute for marking context parameters
///
/// Usage: `#[context]`
///
/// Context parameters are not included in the LLM schema.
/// They must be provided at runtime (for dependency injection).
#[proc_macro_attribute]
pub fn context(_args: TokenStream, input: TokenStream) -> TokenStream {
    // This is a pass-through attribute
    // The #[tool] macro will process it
    input
}

/// Arguments for the #[tool] attribute
#[derive(Debug, Default)]
struct ToolArgs {
    /// Tool name (defaults to function/struct name)
    name: Option<String>,

    /// Tool description
    description: Option<String>,
}

/// Custom parser for tool arguments
impl Parse for ToolArgs {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let mut args = ToolArgs::default();

        // Handle empty attribute like #[tool]
        if input.is_empty() {
            return Err(syn::Error::new(
                proc_macro2::Span::call_site(),
                "description is required: use #[tool(description = \"your description here\")]",
            ));
        }

        while !input.is_empty() {
            // Parse the identifier (name or description)
            let ident: syn::Ident = input.parse().map_err(|e| {
                syn::Error::new(
                    e.span(),
                    format!("Expected 'name' or 'description', got parse error: {}", e),
                )
            })?;

            let ident_str = ident.to_string();

            // Parse the equals sign
            let _: Token![=] = input.parse().map_err(|e| {
                syn::Error::new(
                    e.span(),
                    format!(
                        "Expected '=' after '{}', use syntax: {} = \"...\"",
                        ident_str, ident_str
                    ),
                )
            })?;

            // Parse the string value
            let value: LitStr = input.parse().map_err(|e| {
                syn::Error::new(
                    e.span(),
                    format!(
                        "Expected string literal after '{} =', got parse error: {}",
                        ident_str, e
                    ),
                )
            })?;

            match ident_str.as_str() {
                "name" => args.name = Some(value.value()),
                "description" => args.description = Some(value.value()),
                _ => {
                    return Err(syn::Error::new_spanned(
                        ident,
                        format!(
                            "Unknown attribute '{}'. Expected 'name' or 'description'",
                            ident_str
                        ),
                    ))
                }
            }

            // Handle optional comma
            if input.peek(Token![,]) {
                let _: Token![,] = input.parse()?;
            }
        }

        if args.description.is_none() {
            return Err(syn::Error::new(
                proc_macro2::Span::call_site(),
                "description is required: use #[tool(description = \"your description here\")]",
            ));
        }

        Ok(args)
    }
}

/// The `#[tool]` attribute macro for easy tool definition
///
/// # Usage on Functions
///
/// ```rust,ignore
/// #[tool(description = "Adds two numbers")]
/// fn add_numbers(params: AddParams) -> Result<AddResult, Error> {
///     Ok(AddResult { sum: params.a + params.b })
/// }
/// ```
///
/// # Usage on Structs
///
/// ```rust,ignore
/// #[tool(
///     name = "calculator",
///     description = "Performs arithmetic operations"
/// )]
/// struct Calculator;
///
/// impl Calculator {
///     fn execute(&self, params: CalcParams) -> Result<Value, Error> {
///         // implementation
///     }
/// }
/// ```
///
/// This automatically:
/// - Generates Tool trait implementation
/// - Uses SchemaBasedTool for automatic schema generation
/// - Handles type conversions
/// - Provides error handling
#[proc_macro_attribute]
pub fn tool(args: TokenStream, input: TokenStream) -> TokenStream {
    // Parse the tool arguments
    let tool_args = parse_macro_input!(args as ToolArgs);

    // Try to parse as function first
    if let Ok(func) = syn::parse::<ItemFn>(input.clone()) {
        return expand_tool_function(tool_args, func);
    }

    // Try to parse as struct
    if let Ok(struct_item) = syn::parse::<ItemStruct>(input) {
        return expand_tool_struct(tool_args, struct_item);
    }

    // If neither, return error
    syn::Error::new(
        proc_macro2::Span::call_site(),
        "#[tool] can only be applied to functions or structs",
    )
    .to_compile_error()
    .into()
}

/// Parameter information extracted from function signature
#[derive(Debug, Clone)]
struct ParamInfo {
    name: syn::Ident,
    param_type: Box<syn::Type>,
    description: Option<String>,
    is_individual: bool, // Has #[arg] attribute
}

/// Analyze function parameters flexibly
/// Supports any combination of individual args and struct params in any order
fn analyze_flexible_parameters(
    inputs: &syn::punctuated::Punctuated<syn::FnArg, Token![,]>,
) -> syn::Result<Vec<ParamInfo>> {
    if inputs.is_empty() {
        return Err(syn::Error::new_spanned(
            inputs,
            "Tool function must have at least one parameter",
        ));
    }

    let mut params = Vec::new();

    for input in inputs {
        if let syn::FnArg::Typed(pat_type) = input {
            // Check for #[arg(...)] attribute
            let arg_attr = pat_type
                .attrs
                .iter()
                .find(|attr| attr.path().is_ident("arg"));

            if let syn::Pat::Ident(pat_ident) = &*pat_type.pat {
                let param_name = pat_ident.ident.clone();
                let param_type = pat_type.ty.clone();

                let (is_individual, description) = if let Some(attr) = arg_attr {
                    // Extract description from #[arg(description = "...")]
                    let desc =
                        extract_arg_description(attr).unwrap_or_else(|| param_name.to_string());
                    (true, Some(desc))
                } else {
                    // No #[arg] attribute - treat as struct parameter
                    (false, None)
                };

                params.push(ParamInfo {
                    name: param_name,
                    param_type,
                    description,
                    is_individual,
                });
            }
        }
    }

    Ok(params)
}

/// Extract description from #[arg(description = "...")] attribute
fn extract_arg_description(attr: &syn::Attribute) -> Option<String> {
    if let syn::Meta::List(list) = &attr.meta {
        // Try to parse nested meta
        let mut desc = None;
        let _ = list.parse_nested_meta(|meta| {
            if meta.path.is_ident("description") {
                if let Ok(value) = meta.value() {
                    if let Ok(lit) = value.parse::<LitStr>() {
                        desc = Some(lit.value());
                    }
                }
            }
            Ok(())
        });
        desc
    } else {
        None
    }
}

/// Expand #[tool] on a function
fn expand_tool_function(args: ToolArgs, func: ItemFn) -> TokenStream {
    let func_name = func.sig.ident.clone();
    let tool_name = args.name.unwrap_or_else(|| func_name.to_string());
    let description = args.description.expect("description is required");

    // Analyze all parameters flexibly
    let params = match analyze_flexible_parameters(&func.sig.inputs) {
        Ok(params) => params,
        Err(e) => return TokenStream::from(e.to_compile_error()),
    };

    // Separate individual args from struct params
    let individual_params: Vec<_> = params.iter().filter(|p| p.is_individual).cloned().collect();
    let struct_params: Vec<_> = params
        .iter()
        .filter(|p| !p.is_individual)
        .cloned()
        .collect();

    // Generate the appropriate expansion
    expand_flexible_tool(
        func,
        &func_name,
        &tool_name,
        &description,
        individual_params,
        struct_params,
    )
}

/// Expand tool with flexible parameter combination
/// Supports 0-n individual args + 0-n struct params in any order
fn expand_flexible_tool(
    func: ItemFn,
    func_name: &syn::Ident,
    tool_name: &str,
    description: &str,
    individual_params: Vec<ParamInfo>,
    struct_params: Vec<ParamInfo>,
) -> TokenStream {
    let pascal_name = func_name.to_string().to_case(Case::Pascal);
    let struct_name = syn::Ident::new(&format!("{}Tool", pascal_name), func_name.span());

    // Case 1: Only struct params (1 or more)
    if individual_params.is_empty() && struct_params.len() == 1 {
        // Simple case: single struct parameter (original mode)
        return expand_single_struct_tool(
            func,
            func_name,
            tool_name,
            description,
            struct_params[0].param_type.clone(),
        );
    }

    // Case 2: Only individual params (1 or more)
    if struct_params.is_empty() && !individual_params.is_empty() {
        return expand_individual_params_tool(
            func,
            func_name,
            tool_name,
            description,
            individual_params,
        );
    }

    // Case 3: Mixed (both individual and struct params)
    // Case 4: Multiple struct params
    // For now, these advanced cases are not fully implemented
    // Show a helpful error message

    if !individual_params.is_empty() && !struct_params.is_empty() {
        return syn::Error::new_spanned(
            &func.sig.inputs,
            "Mixed parameters (individual + struct) not yet supported.\n\
             Use either all individual params with #[arg(...)] OR single struct param.",
        )
        .to_compile_error()
        .into();
    }

    if struct_params.len() > 1 {
        return syn::Error::new_spanned(
            &func.sig.inputs,
            "Multiple struct parameters not yet supported.\n\
             Combine all params into a single struct.",
        )
        .to_compile_error()
        .into();
    }

    // Fallback error
    syn::Error::new_spanned(&func.sig.inputs, "Unable to determine parameter mode")
        .to_compile_error()
        .into()
}

/// Expand tool with single struct parameter (original mode)
fn expand_single_struct_tool(
    func: ItemFn,
    func_name: &syn::Ident,
    tool_name: &str,
    description: &str,
    param_type: Box<syn::Type>,
) -> TokenStream {
    // Extract return type (for future validation)
    let _return_type = match &func.sig.output {
        syn::ReturnType::Type(_, ty) => ty,
        _ => {
            return syn::Error::new_spanned(&func.sig.output, "Tool function must return a Result")
                .to_compile_error()
                .into();
        }
    };

    // Generate struct name for the tool
    // Convert snake_case function name to PascalCase + Tool suffix
    let pascal_name = func_name.to_string().to_case(Case::Pascal);
    let struct_name = syn::Ident::new(&format!("{}Tool", pascal_name), func_name.span());

    // Generate the implementation
    let expanded = quote! {
        #func

        // Generate a struct to implement the tool trait
        pub struct #struct_name;

        impl ::rsllm::tools::SchemaBasedTool for #struct_name {
            type Params = #param_type;

            fn name(&self) -> &str {
                #tool_name
            }

            fn description(&self) -> &str {
                #description
            }

            fn execute_typed(
                &self,
                params: Self::Params,
            ) -> Result<::serde_json::Value, Box<dyn std::error::Error + Send + Sync>> {
                // Call the original function
                let result = #func_name(params)?;

                // Convert result to JSON
                ::serde_json::to_value(&result)
                    .map_err(|e| Box::new(e) as Box<dyn std::error::Error + Send + Sync>)
            }
        }
    };

    TokenStream::from(expanded)
}

/// Expand tool with individual parameters only
fn expand_individual_params_tool(
    func: ItemFn,
    func_name: &syn::Ident,
    tool_name: &str,
    description: &str,
    params: Vec<ParamInfo>,
) -> TokenStream {
    // Generate a params struct from individual parameters
    let pascal_name = func_name.to_string().to_case(Case::Pascal);
    let params_struct_name = syn::Ident::new(&format!("{}Params", pascal_name), func_name.span());
    let struct_name = syn::Ident::new(&format!("{}Tool", pascal_name), func_name.span());

    // Build struct fields with doc comments
    let param_fields = params.iter().map(|p| {
        let name = &p.name;
        let ty = &p.param_type;
        let doc = p.description.as_deref().unwrap_or("");
        quote! {
            #[doc = #doc]
            pub #name: #ty
        }
    });

    // Build function call arguments in the original order
    let call_args = params.iter().map(|p| {
        let name = &p.name;
        quote! { generated_params.#name }
    });

    let expanded = quote! {
        #func

        // Auto-generated params struct
        #[derive(::schemars::JsonSchema, ::serde::Serialize, ::serde::Deserialize)]
        pub struct #params_struct_name {
            #(#param_fields),*
        }

        // Generate tool struct
        pub struct #struct_name;

        impl ::rsllm::tools::SchemaBasedTool for #struct_name {
            type Params = #params_struct_name;

            fn name(&self) -> &str {
                #tool_name
            }

            fn description(&self) -> &str {
                #description
            }

            fn execute_typed(
                &self,
                generated_params: Self::Params,
            ) -> Result<::serde_json::Value, Box<dyn std::error::Error + Send + Sync>> {
                // Call the original function with unpacked params
                let result = #func_name(#(#call_args),*)?;

                // Convert result to JSON
                ::serde_json::to_value(&result)
                    .map_err(|e| Box::new(e) as Box<dyn std::error::Error + Send + Sync>)
            }
        }
    };

    TokenStream::from(expanded)
}

/// Expand #[tool] on a struct
fn expand_tool_struct(args: ToolArgs, struct_item: ItemStruct) -> TokenStream {
    let struct_name = &struct_item.ident;
    let tool_name = args
        .name
        .unwrap_or_else(|| struct_name.to_string().to_lowercase());
    let description = args.description.expect("description is required");

    // For struct, we expect the user to implement an execute method manually
    // This macro just generates the Tool trait boilerplate

    let expanded = quote! {
        #struct_item

        // Note: User must implement execute_typed manually
        // This just provides the name and description
        impl #struct_name {
            pub fn tool_name() -> &'static str {
                #tool_name
            }

            pub fn tool_description() -> &'static str {
                #description
            }
        }
    };

    TokenStream::from(expanded)
}
