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
                    format!("Expected '=' after '{}', use syntax: {} = \"...\"", ident_str, ident_str),
                )
            })?;

            // Parse the string value
            let value: LitStr = input.parse().map_err(|e| {
                syn::Error::new(
                    e.span(),
                    format!("Expected string literal after '{} =', got parse error: {}", ident_str, e),
                )
            })?;

            match ident_str.as_str() {
                "name" => args.name = Some(value.value()),
                "description" => args.description = Some(value.value()),
                _ => {
                    return Err(syn::Error::new_spanned(
                        ident,
                        format!("Unknown attribute '{}'. Expected 'name' or 'description'", ident_str),
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

/// Detect the parameter mode from function signature
enum ParameterMode {
    /// Single struct parameter: fn tool(params: MyParams)
    SingleStruct(Box<syn::Type>),
    /// Individual parameters: fn tool(#[arg(...)] a: i32, #[arg(...)] b: String)
    Individual(Vec<(syn::Ident, Box<syn::Type>, Option<String>)>),
    /// Mixed: fn tool(#[context] ctx: Context, params: MyParams)
    Mixed {
        context_params: Vec<(syn::Ident, Box<syn::Type>)>,
        struct_param: Box<syn::Type>,
    },
}

fn analyze_parameters(inputs: &syn::punctuated::Punctuated<syn::FnArg, Token![,]>) -> syn::Result<ParameterMode> {
    if inputs.is_empty() {
        return Err(syn::Error::new_spanned(
            inputs,
            "Tool function must have at least one parameter",
        ));
    }

    let mut regular_params = Vec::new();
    let mut context_params = Vec::new();

    for input in inputs {
        if let syn::FnArg::Typed(pat_type) = input {
            // Check for #[arg(...)] or #[context] attributes
            let has_arg_attr = pat_type.attrs.iter().any(|attr| attr.path().is_ident("arg"));
            let has_context_attr = pat_type.attrs.iter().any(|attr| attr.path().is_ident("context"));

            if let syn::Pat::Ident(pat_ident) = &*pat_type.pat {
                let param_name = pat_ident.ident.clone();
                let param_type = pat_type.ty.clone();

                if has_context_attr {
                    context_params.push((param_name, param_type));
                } else if has_arg_attr {
                    // Extract description from #[arg(description = "...")]
                    let description = pat_type.attrs.iter()
                        .find(|attr| attr.path().is_ident("arg"))
                        .and_then(|attr| {
                            // Try to extract description from the attribute
                            if let syn::Meta::List(_list) = &attr.meta {
                                // Simple extraction - just get the string
                                Some("Individual parameter".to_string())
                            } else {
                                None
                            }
                        });
                    regular_params.push((param_name, param_type, description));
                } else {
                    // No attributes - regular param (assume it's the struct)
                    regular_params.push((param_name, param_type, None));
                }
            }
        }
    }

    // Determine mode
    if context_params.is_empty() && regular_params.len() == 1 {
        // Single struct mode
        Ok(ParameterMode::SingleStruct(regular_params[0].1.clone()))
    } else if !context_params.is_empty() && regular_params.len() == 1 {
        // Mixed mode
        Ok(ParameterMode::Mixed {
            context_params,
            struct_param: regular_params[0].1.clone(),
        })
    } else if context_params.is_empty() && regular_params.iter().all(|(_, _, desc)| desc.is_some() || regular_params.len() > 1) {
        // Individual parameters mode
        Ok(ParameterMode::Individual(regular_params))
    } else {
        Err(syn::Error::new_spanned(
            inputs,
            "Ambiguous parameter mode. Use either:\n\
             1. Single struct: fn tool(params: MyParams)\n\
             2. Individual params: fn tool(#[arg(...)] a: i32, #[arg(...)] b: String)\n\
             3. Mixed: fn tool(#[context] ctx: Ctx, params: MyParams)",
        ))
    }
}

/// Expand #[tool] on a function
fn expand_tool_function(args: ToolArgs, func: ItemFn) -> TokenStream {
    let func_name = func.sig.ident.clone();
    let tool_name = args.name.unwrap_or_else(|| func_name.to_string());
    let description = args.description.expect("description is required");

    // Analyze parameters to determine mode
    let param_mode = match analyze_parameters(&func.sig.inputs) {
        Ok(mode) => mode,
        Err(e) => return TokenStream::from(e.to_compile_error()),
    };

    // Handle different parameter modes
    match param_mode {
        ParameterMode::SingleStruct(param_type) => {
            expand_single_struct_tool(func, &func_name, &tool_name, &description, param_type)
        }
        ParameterMode::Individual(params) => {
            expand_individual_params_tool(func, &func_name, &tool_name, &description, params)
        }
        ParameterMode::Mixed { context_params, struct_param } => {
            expand_mixed_params_tool(func, &func_name, &tool_name, &description, context_params, struct_param)
        }
    }
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
            return syn::Error::new_spanned(
                &func.sig.output,
                "Tool function must return a Result",
            )
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

/// Expand tool with individual parameters
fn expand_individual_params_tool(
    func: ItemFn,
    func_name: &syn::Ident,
    tool_name: &str,
    description: &str,
    params: Vec<(syn::Ident, Box<syn::Type>, Option<String>)>,
) -> TokenStream {
    // Generate a params struct from individual parameters
    let pascal_name = func_name.to_string().to_case(Case::Pascal);
    let params_struct_name = syn::Ident::new(&format!("{}Params", pascal_name), func_name.span());
    let struct_name = syn::Ident::new(&format!("{}Tool", pascal_name), func_name.span());

    // Build struct fields
    let param_fields = params.iter().map(|(name, ty, _desc)| {
        quote! { pub #name: #ty }
    });

    // Build function call arguments
    let call_args = params.iter().map(|(name, _, _)| {
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

/// Expand tool with mixed mode (context + struct params)
fn expand_mixed_params_tool(
    func: ItemFn,
    _func_name: &syn::Ident,
    _tool_name: &str,
    _description: &str,
    _context_params: Vec<(syn::Ident, Box<syn::Type>)>,
    _struct_param: Box<syn::Type>,
) -> TokenStream {
    // For mixed mode, the schema only includes the struct param
    // Context params are not exposed to the LLM

    // Note: In mixed mode, context params need to be provided at execution time
    // This requires a different approach - for now, show a helpful error

    let expanded = quote! {
        compile_error!(
            "Mixed mode (context params + struct) not fully implemented yet.\n\
             Context parameters require runtime injection.\n\
             For now, use either:\n\
             1. Single struct: fn tool(params: MyParams)\n\
             2. Individual params: fn tool(#[arg(...)] a: i32, #[arg(...)] b: String)\n\
             \n\
             Mixed mode coming in next update!"
        );

        #func
    };

    TokenStream::from(expanded)
}

/// Expand #[tool] on a struct
fn expand_tool_struct(args: ToolArgs, struct_item: ItemStruct) -> TokenStream {
    let struct_name = &struct_item.ident;
    let tool_name = args.name.unwrap_or_else(|| struct_name.to_string().to_lowercase());
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
