//! Procedural macros for RSLLM tool calling
//!
//! This crate provides the `#[tool]` attribute macro for easy tool definition.

use convert_case::{Case, Casing};
use proc_macro::TokenStream;
use quote::quote;
use syn::{
    parse::{Parse, ParseStream},
    parse_macro_input, ItemFn, ItemStruct, LitStr, Token,
};

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

/// Expand #[tool] on a function
fn expand_tool_function(args: ToolArgs, func: ItemFn) -> TokenStream {
    let func_name = &func.sig.ident;
    let tool_name = args.name.unwrap_or_else(|| func_name.to_string());
    let description = args.description.expect("description is required");

    // Extract parameter type from function signature
    let param_type = match func.sig.inputs.first() {
        Some(syn::FnArg::Typed(pat_type)) => &pat_type.ty,
        _ => {
            return syn::Error::new_spanned(
                &func.sig.inputs,
                "Tool function must have exactly one parameter",
            )
            .to_compile_error()
            .into();
        }
    };

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
