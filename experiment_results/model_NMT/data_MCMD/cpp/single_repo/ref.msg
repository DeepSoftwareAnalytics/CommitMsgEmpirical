[ CS ] Suppress a no - asserts warning
[ Clang importer ] After stripping a prefix , lowercase initialisms as well .
[ swiftc ( 62 vs . 5600 ) ] Add crasher in swift : : constraints : : ConstraintGraph : : lookupNode ( . . . )
[ swiftc ( 51 vs . 5592 ) ] Add crasher in swift : : ProtocolType : : canonicalizeProtocols
[ SE - 0258 ] Rename backing storage property to _foo .
[ stdlib ] Remove redundant + ( _ : _ : ) overload in RangeReplaceableCollectionType
[ GSB ] Distinguish archetype resolution kind for getNestedType ( ) .
SILGen : Add test that ' defer ' bodies can capture generic context .
Source annotation : fix a crash on var decls with propagated types
Merge remote - tracking branch ' origin / master ' into master - llvm - swift5 - transition
Merge pull request from drodriguez / windows - long - tmp - path
[ sil ] Rename SILSuccessorIterator - > SILSuccessor : : pred_iterator .
Roll - back part of master - next change 63a9943a46
Revert " Simplify resolveLocatorToDecl to return a ConcreteDeclRef instead of "
Merge remote - tracking branch ' origin / master ' into master - next
Merge remote - tracking branch ' origin / master ' into master - rebranch
Merge remote - tracking branch ' origin / master ' into master - next
Diagnose exclusivity in the presence of coroutines .
Merge pull request from compnerd / fork
[ capture - promotion ] Change isNonEscapingUse to use a visitor .
Avoid uninitialized " random " bytes on Windows .
Merge remote - tracking branch ' origin / master ' into master - next
Merge pull request from jrose - apple / mangle - typealiases - better
Restore AllCases Default
Merge pull request from AndrewSB / patch - 3
Postpone type checking of stdlib bridge functions until we encounter a declaration that will require one . ( rdar : / / problem / 16520213 )
Merge pull request from nielsandriesse / patch - 16
Limit exhaustivity checking when weird things happen with ' @ unknown '
Merge pull request from compnerd / r352827
Merge remote - tracking branch ' origin / master ' into master - next
Merge remote - tracking branch ' origin / master ' into master - next
Precedence group declarations are only valid at file scope .
[ APINotes ] One more 0 - valued option set case . ( )
Merge pull request from jckarter / opaque - overload - checking
Merge pull request from gregomni / typealias
sink SILDeclRef ' s down into emitGetAccessor / emitSetAccessor , which are
Merge pull request from apple / compnerd - swiftpm - pd - libs
[ Driver ] Hook up - emit - dependencies option , but don ' t do anything with it yet .
Merge pull request from martinboehme / cxx - exceptions - doc
[ Omit needless words ] Split at a preposition before " of " , if one exists .
Don ' t use switch instructions
Merge remote - tracking branch ' origin / master ' into master - next
Merge remote - tracking branch ' origin / master ' into master - next
Merge pull request from mikeash / fix - test - generic - casts - objc
Merge pull request from meg - gupta / cseossa
Remove stale - enable - sil - ownership flag from test
Merge pull request from aschwaighofer / irgen_fix_protocol_any_object_to_protocol_composition_with_superclass
[ Constraint graph ] Restrict connected components to requested type variables .
SymbolGraph : Don ' t unconditionally add edge targets to the graph
Merge pull request from bob - wilson / fix - linetable - test
Merge pull request from devincoughlin / more - exclusivity - tests
Merge remote - tracking branch ' origin / master ' into master - next
Add an - enable - block - bridging frontend option we can hide disruptive changes to block bridging behind .
When solving for type variable bindings , protect against trying the same ( potentially unwrapped ) binding twice .
[ stdlib ] String : RangeReplaceableCollection & BidirectionalCollection ( )
Keyword argument QoI : use typo correction to help match keyword arguments to parameters .
Merge remote - tracking branch ' origin / master ' into master - next
Implement identity comparison correctly for begin_access / end_access .
Merge pull request from compnerd / r357552
Merge pull request from compnerd / slash - and - test
[ ConstraintSystem ] Don ' t check ` - disable - availability - checking `
Merge remote - tracking branch ' origin / master ' into master - next
Private method should be for non - Windows only
benchmarks : add another benchmark for appending multiple array elements .
Merge pull request from gottesmm / pr - 4bb26d5be9b2e36ec5a1897a5d682904f89ddab4
[ Serialization ] Don ' t look at extension members when resolving XREFs .
Overloads for + / - on Strideable for Swift 3 compatibility mode
Merge pull request from xymus / silence - outdated - pch - error
Merge remote - tracking branch ' origin / master ' into master - next
Merge pull request from compnerd / externalities
[ silgen ] Add { ManagedValue , RValue } : : ensurePlusOne ( . . . ) .
Sema : Stop finalizing struct and enum members
Merge remote - tracking branch ' origin / master ' into master - rebranch
[ Type checker ] Tolerate missing generic environments better .
increase the # bits for DeclContextKind ' s to 3 bits .
[ silgen ] Add ArgumentSource : : Kind : : DelayedBorrowedRValue .
SILGen : Clean up an awkward use of AbstractFunctionDecl : : getImplicitSelfDecl ( )
James Campbell pointed out on swift - evolution that the [ : ] literal syntax
Merge pull request from slavapestov / noescape - by - default - typealias - fix
[ swiftc ( 136 vs . 5221 ) ] Add crasher in swift : : ASTVisitor
stdlib / Array : add a test that arrays of non - Equatable elements are not
Merge remote - tracking branch ' origin / master ' into master - next
Merge remote - tracking branch ' origin / master ' into master - rebranch
Merge pull request from practicalswift / swiftc - 28396 - swift - lowering - silgenfunction - emitclosurevalue
SourceEntityWalker : Add a parameter to visitDeclReference ( ) describing the kind of the reference under visit . NFC
IUO : Use new ImportedType class rather than std : : pair < Type , bool > .
Improve strange - interpolation test
Merge pull request from benlangmuir / cc - enum - leading - dot
Merge remote - tracking branch ' origin / master ' into master - next
Merge remote - tracking branch ' origin / master ' into master - next
[ ls - opts ] Round to nearest power of 2 when reserving buckets for BBToBBIDMap
Merge pull request from hughbe / silgenexpr - stdmove
Release notes : - emit - llvm and - triple have been deprecated .
build : support aliases for TARGET_SDKS
Merge pull request from lorentey / disable - string - literal - concat - da
[ Printer ] add sil - disable - ast - dump to disable dumping top AST decls .
Merge pull request from compnerd / analytical - methods
Remove dead code in SILGenProlog . cpp
Merge pull request from artemcm / ImplOnlyPropertyWrappers
Test nonempty - brace - in - brace
Annotate two crash tests that don ' t crash anymore
Use hasArchetype directly on SILType rather than going through Type .
Addressing comments .
Merge remote - tracking branch ' origin / master ' into master - next
SIL : Type lowering for imported ObjC async decls .
add some comments in AliasAnalysis and EscapeAnalysis
Mangling : Fix bug in the logic for dropping same - type constraints
docs / CMakeLists . txt : documentation
[ Test ] Add an executable test
Merge remote - tracking branch ' origin / master ' into master - next
Merge pull request from lancep / comprehensiveArrayTests
Create a fast path for release silargument .
IUO : Set the IUO attribute on ParamDecls for accessors arguments .
rearrange testcase .
[ stdlib ] Array bridging specification tweaks
Mark some more things public . . . and mark ArrayType internal again !
Concurrency : Change Unsafe [ Throwing ] Continuation < T > to store a RawUnsafeContinuation
[ ASTMangler ] Don ' t verify the USR mangling for invalid decls
SIL optimizer : Support begin_cow_mutation and end_cow_mutation in some optimizations .
Merge pull request from apple / revert - 17668 - master
SIL : Capture generic parameters through nested closures .
Merge remote - tracking branch ' origin / master ' into master - next
Use # elif
Sink getArgParamPatterns ( ) and getBodyParamPatterns ( ) into AbstractFunctionDecl
Add support for default initialized values with a new ZeroValue instruction .
build : enable cross - compilation on Windows
[ Stats ] Permit negative counters and deltas .
[ ABI ] Add default associated type witnesses to resilient protocols .
[ TypeChecker ] Fix dynamic member declaration validation to check labels
Debug Info : Generate imported entities for implicitly imported parent modules .
[ TBDGen ] Emit case symbols only if enum is resilient and some other fixes
Parse : Fix recursive diagnostic corner - case with invalid UTF8 in source file
update test for a diagnostic change .
Add a verifier to check that the protocol conformances provided by
CMake : clangCodeGen dependends on LLVMDebugInfoDWARF
straighten out identifiers in diagnostics a bit .
ClangImporter : adjust for SVN r345637
[ ClangImporter ] Respect SwiftImportAsAccessors in APINotes . ( )
[ docs ] " Retroactive protocol refinement " in generics manifesto .
Merge remote - tracking branch ' origin / main ' into next
Merge remote - tracking branch ' origin / master ' into master - next
SIL Serialization : handle more SILInstructions
IRGen : Fix ' super . ' calls to objc - dispatched static methods from generic classes
[ sil - combine ] Disable unchecked_take_enum_data_addr promotion to unchecked_enum_data when the enum type is address only .
Merge remote - tracking branch ' origin / master ' into master - next
SILTypeList was removed . . . Removing LLVM_DECLARE_TYPE_ALIGNMENT for it .
[ gardening ] Fixup a couple of doxygen comments .
Merge remote - tracking branch ' origin / master ' into master - next
[ Gardening ] Made FSOEnableGenerics option static .
Merge pull request from atrick / fix
[ CSBindings ] Add ` finalize ` method to ` PotentialBindings `
AST : Remove ArgumentShuffleExpr
[ Tools ] Add a reminder to run the tool as root if the target can ' t be inspected .
reindent handleErrorAndSupplyMissingMember
[ pred - memopt ] Add convenience begin / end borrow operation on SILBuilder .
ClangImporter : Bump libpthread SDK Epoch to 1
Made LazyLoaderArray ' s length type fit in a 32 - bit pointer .
Merge remote - tracking branch ' origin / master ' into master - next
SILGen : Move some code around in pattern emission
remove some peephole logic that is trying to turn a copyaddr into its component
Merge pull request from LucianoPAlmeida / SR - 13262 - where - needs - bool
Improve casts analysis for bridged types .
[ swiftc ( 60 vs . 5600 ) ] Add crasher in swift : : ProtocolConformance : : getTypeWitnessAndDecl
[ SourceKit ] Report container type USRs in the response to cursor - info request .
Merge pull request from modocache / sr - 1738 - add - swift - library - shared - static - args
Test restrictions on objc_runtime_visible classes .
Merge remote - tracking branch ' origin / master ' into master - next
Typo : consequtive - > consecutive
[ gardening ] Remove typedef that was used in only one place .
Minor rearrangement .
tests : re - add a removed test
Merge remote - tracking branch ' origin / master ' into master - next
Merge remote - tracking branch ' origin / master ' into master - next
Merge the dense maps for the representative of a type variable and the
Merge pull request from DougGregor / api - notes - change - types
Merge remote - tracking branch ' origin / master ' into master - rebranch
[ stdlib ] [ NFC ] Cosmetics
Merge remote - tracking branch ' origin / master ' into master - next
docs : diagram of current Array implementation in stdlib .
Merge pull request from swiftix / generic - specialization - fixes
My diagnstics change agitated a validation test , grr .
Merge remote - tracking branch ' origin / master ' into master - next
Merge remote - tracking branch ' origin / master ' into master - next
CoreGraphics overlay : Comment that the @ transparent definitions really want to be @ fragile .
Zap some dead code attempting to deal with broken inherited conformances . NFC
Merge pull request from benlangmuir / bump - ver - default
Merge remote - tracking branch ' origin / master ' into master - next
Merge remote - tracking branch ' origin / master ' into master - next
AST : attempt to rewrite the expression to be more legible
[ CodeCompletion ] Merge shouldUseFunctionReference to addCompoundFunctionName
[ SIL ] Ensure that all loop branches have valid locations .
[ ClangImporter ] Add a bunch of non - null assertions .
Fix typo , found by looking through commits .
Merge pull request from ahoppen / sema - enabled - notificaiton
IDE : Remove hasAccess ( ) checks
docs / Windows : Format and clarify documentation
Disable test_superclass_methods . swift and test_superclass_properties . swift
When importing external enum types , delay the creation of implicit members and their associated support functions until after the types are actually referenced .
Move the " is there a call to super . init ? " check into a ConstructorDecl predicate .
WinSDK : add CW_USEDEFAULT to the SDK overlay
Merge pull request from slavapestov / fixed - layout - classes
add SILBase . cpp
[ CS ] NFC : Inline getSummaryFlagsForPathElement
Merge pull request from compnerd / windows - tls - cc
[ DiagnosticVerifier ] Add support for asserting presence of edu notes
Merge pull request from compnerd / WinSock
Merge remote - tracking branch ' origin / master ' into master - next
[ Frontend ] Remove parseAndTypeCheckMainFileUpTo
Merge pull request from xedin / solver - get - params
Merge pull request from aschwaighofer / codesign_evolution_tests
Update for llvm StringRef string conversion
Merge pull request from slavapestov / field - offset - globals - are - too - resilient
[ CS ] NFC : Remove OverloadChoiceKind : : BaseType
[ FixCode ] Avoid removing @ IB * attributes
[ CSStep ] Make sure that components are sorter before execution
implement skeleton of extension body parsing .
[ sil - bug - reducer ] SILTools . sil_func_extractor should return sil_func_extractor , not sil_opt .
stdlib : don ' t use doc comment markup for non - doc comments
tests : workaround for rdar : / / problem / 24330686
AST : We ' re not going to be using the Resilience enum , NFC
Simplify handling of lookup results into DynamicLookup .
Add - sil - inline - never - functions flag .
[ IRGen ] Allow constructing UniversalLinkageInfo without an IRGenModule .
Sema : Ban single - element tuple expressions
Tweak testcase to work with release and debug stdlibs .
CMake : adjust indentation for consistency
Merge remote - tracking branch ' origin / master ' into master - next
Merge remote - tracking branch ' origin / master ' into master - next
Remove SharedTimer Infrastructure
Follow - up change to PR30186
[ Python ] Remove unused code .
[ SubstitutionMap ] Eliminate the parent map and handle archetype conformances .
Merge pull request from moiseev / intents - warnings
Revert " Fix two problems with implicit conversions : - rdar : / / problem / 16776273 , wherein conversions between nil and . None were permitted due to an implicit conversion between nil and COpaquePointer . - rdar : / / problem / 16877526 , where we needed to add new equality overloads to handle conversions between nil and . None given the supression of user conversions . ( Thanks to Ted for the overloads and test . ) "
SILGen : Emit semantic - to - storage conversions in implicit struct ctors .
WIP
[ SR - 3373 ] clarify error message when trying to use instance member statically
Change REQUIRES arguments to work with recent LLVM versions
remove the dead ' TypeConverter : : getMethodSelfType ' method , which
Add conditional logic for including benchmarks .
[ AST ] NFC : Repack misc TupleExpr bits
[ silgen ] Make ensurePlusOne emit copies even when guaranteed normal arguments is not enabled .
[ Dependency scanning ] Cope with inferred module maps
Merge remote - tracking branch ' origin / master ' into master - next
More executable_test for the test directory
Restore const - ness of TypeRefs
Prevent an exclusivity benchmark from being optimized too much .
Merge pull request from mikeash / add - swiftprivate - darwin - dependency
Merge pull request from drodriguez / android - mark - executable - mandatory - inlining
Add ' Sema ' to Lexicon
[ AST ] Explicitly track things marked __owned .
Move a section in release notes
Misc spot fixes
[ stdlib ] Some cleanup enabled by _alwaysEmitIntoClient .
Sema : If a property type was inferred , copy it to the accessor before checking generic signature
Merge pull request from nkcsgexi / syntax - disambiguity
[ ownership ] Move ownership lowering past SROA .
[ CodeCompletion ] Eliminate CodeCompletionCallbacks : : completeExpr ( )
IRGen : Work around RemoteMirror bug generating reflection info for empty builtin types .
Resolve comments from last code review
Merge remote - tracking branch ' origin / master ' into master - next
Fix MSVC / GCC compilation failures resulting from new refcount representation
[ sil - combine ] Refactor optimizeBuiltinCanBeObjCClass from InstSimplifer - > SILCombine since it creates instructions .
Merge pull request from linux - on - ibm - z / s390x - keypath - tuple - fix
[ CSSolver ] add debug logging to ` solveForCodeCompletion `
Tuple types with a single element are now valid . var x : ( ( ( int ) ) ) ; is now a triply nested tuple . To make this work , values are allowed to convert to tuples with one element .
Teach mandatory inlining how to use closure deletion infrastructure .
[ AST ] Shrink CaptureInfo down to a PointerIntPair ( )
[ GSB ] Rework construction of nested types .
Remove Swift ' s copy of llvm : : interleave
Placate - Wunreachable - code by using # else for alternative code .
[ Sema ] Gardening : Remove unused struct [ NFC ]
DI : Correctly flag escaping uses of ' self ' when ' self ' is passed as an argument to self / super . init
Merge pull request from itaiferber / datecomponents - use - encodeifpresent
Merge remote - tracking branch ' origin / master ' into master - next
move the order of . swift files . This order crashes the compiler .
SILGen : Catch up with new Expr nodes .
[ docs ] Add comments clarifying roles of Type . h and Types . h .
[ Exclusivity ] Make helper functions to static . NFC .
[ windows ] Fix Python tests in Windows .
[ semantic - arc ] Add the ability to at the command line run a subset of semantic - arc optimizations .
[ QoI ] improve diagnostics for operator declarations ; unify parsing code
Revert " Simplify performMemberLookup ( ) "
Merge remote - tracking branch ' origin / master ' into master - next
[ CSBindings ] Infer l - value type for object of ` ForceValueExpr ` only if var allows it
Clang importer : use lookup table version number in the module file extension hash .
Add sources for swift - stdlib - tool .
Merge pull request from gottesmm / fix_construction_during_invalidation
Revert " Enable cmark smart punctuation in doc comments "
[ libSyntax ] Add test variants for building the syntax map via libSyntax
Merge remote - tracking branch ' origin / master ' into master - next
[ benchmark ] Extract setup : MapReduceClass ( Small )
Merge pull request from gottesmm / wireopt_optfunc_reducers_to_randomsearch
Sema : Only warn about @ inlinable implying @ usableFromInline in - swift - version 4 . 2
When we detect a property / subscript that has a setter without a getter , create a dummy
Fix - Wnon - virtual - dtor warning .
Merge remote - tracking branch ' origin / master ' into master - next
Merge remote - tracking branch ' origin / master ' into master - next
Remove redundant literal constraints on a type variable .
libSyntax : rename KeepTokensInSourceFile to KeepSyntaxInfoInSourceFile .
[ index ] Fix infinite loop while looking at superclasses
Enable pointer conversions in the test that needs it .
[ Accelerate ] vImage , vDSP , vForce : Describe availability
Merge pull request from bob - wilson / llvm - r332240
Merge pull request from DougGregor / assoc - conformance - default - witnesses
[ ownership ] Add support for handling address_to_pointer to LoadBorrowInvalidationChecker .
Reenable temporarily disabled test .
tests : enable a few tests on Linux , u - opt
Add - debug - info - format = [ dwarf | codeview ] option ( )
Merge pull request from DougGregor / de - gyb - interpolation
Merge pull request from DougGregor / type - checker - simplify - error - type
[ AST ] Add support for multiple trailing closures to the parser / expressions
Handle ` nil ` with any number of parentheses
Merge remote - tracking branch ' origin / master ' into master - next
Merge remote - tracking branch ' origin / master ' into master - rebranch
Turn SILLocation out of typedef and into a class .
Merge remote - tracking branch ' origin / master ' into master - next
Merge pull request from gottesmm / add_cmake_comment
Reformat to reduce indentation .
Merge pull request from nkcsgexi / parser - delayed - control
Merge pull request from DougGregor / protocol - symbolic - reference
Merge remote - tracking branch ' origin / master ' into master - llvm - swift5 - transition
Temporarily remove the global min ( a , b ) and max ( a , b ) SIMD functions . ( )
Merge remote - tracking branch ' origin / master ' into master - next
IRGen : Fix type layout cache leak
Fix a couple of places in the IRGen where we still don ' t use interface types
Merge pull request from tbkka / tbkka / dynamicCastRework - Specification - c
Merge pull request from practicalswift / fix - typos - 20160113b
AST : Introduce ParameterList : : getParams ( )
Address the simple issues raised by Graydon and Huon
Merge pull request from huonw / symbol - list - 3
[ Concurrency ] Allow ActorIsolation in diagnostic messages .
LangRef : Describe the new pattern forms .
Merge pull request from DougGregor / more - delayed - body - parsing
Merge remote - tracking branch ' origin / master ' into master - next
Merge pull request from gottesmm / pr - 99f20d06f6c88f3ed5fa775959f12f6b5eb5f1b8
Omit needless words : " object " is a terrible verb
[ Runtime ] Clean up metadata allocation tracking .
Merge remote - tracking branch ' origin / main ' into next
Merge pull request from benlangmuir / libcxx - less
Sema : Merge validateAttributes ( ) into checkDeclAttributes ( )
Lex $ notAllDigits as an identifier and diagnose it in the lexer
Merge remote - tracking branch ' origin / master ' into master - next
Allow shadowing of unavailable members
The runtime function swift_autoDiffCreateLinearMapContext was recently added
Parse trailing closure productions as expr - postfix .
Don ' t print double ` $ ` signs
[ stats ] Add UnifiedStatsReporter : : FrontendStatsTracer
[ silgen ] Pass foreignSelf to applyPartiallyAppliedSuperMethod by value instead of by ref .
[ CoreGraphics Renaming ] Overhaul CG apinotes
Remove other color checks from the AST dump that are never used .
WIP executable
Merge remote - tracking branch ' origin / master ' into master - rebranch
Merge remote - tracking branch ' origin / master ' into master - next
[ gardening ] Remove asserts from fixed crashers .
Merge remote - tracking branch ' origin / master ' into master - next
Merge remote - tracking branch ' origin / master ' into master - next
ASTMangler : Mangle sugared types for the debugger
Merge remote - tracking branch ' origin / master ' into master - rebranch
[ SourceKit ] Add test case for crash triggered in swift : : TypeChecker : : typeCheckFunctionBodyUntil ( swift : : FuncDecl * , swift : : SourceLoc )
Added " parent type " to the lexicon
Merge pull request from DougGregor / type - checker - limit - objc
Add SILInstruction : : moveAfter ( SILInstruction * ) .
Merge pull request from practicalswift / sourcekit - 084 - swift - parser - consumedecl
[ Stdlib ] Add notes to lazy collection doc comments about complexity
Write out filemaps for supplementary outputs .
[ IDE ] [ DocSupport ] Fix DocInfo missing decls when generated for clang submodules .
[ migrator ] Migrate UIApplicationMain
KeyPaths : Ensure mutation accesses are ended in the correct order .
Merge remote - tracking branch ' origin / master ' into master - next
Fix XFAILed testcase by re - arranging CHECK lines .
[ strip - ST ] Disable runtime stack trace dumping on Darwin when asserts are disabled .
Merge pull request from xedin / rdar - 39040593
Merge pull request from bitjammer / rdar - 30404063 - pcr
Merge remote - tracking branch ' origin / master ' into master - next
DI : Remove unnecessary assertion
Merge remote - tracking branch ' origin / master ' into master - next
Merge pull request from apple / update - checkout - config
[ docs ] LibraryEvolution : Make code snippets more concrete , take 2 .
Front - end : EmitSyntax action should use the syntax tree generated from parser . ( )
build : remove ` OBJECT_LIBRARY ` from ` _add_swift_host_library_single `
[ migrator ] Revert " migrator : avoid inserting base name while renaming if users ' member access doesn ' t specify the base name . rdar : / / 40373279 "
XFAIL a flaky test
Helper class for visiting basic blocks in dominance order .
[ os_log ] [ stdlib - private ] Add tests for checking the correctness of
Pass PrintOptions by const reference instead of by value .
[ GSB ] Use addConditionalRequirements for all conditional - requirement - adding .
Merge pull request from dcci / missing - return
Fix test case
[ Scope map ] Cope with pattern binding accessor / initializer children in any order .
Merge pull request from compnerd / uuid
IRGen : Fix the address type of empty boxes .
Given selector - data arrays a more semantic name .
Fix a typo in $ { NEW } variable in Post - processing tools for diagnostics section
Swap InputFileKind for ParseInputMode
Merge remote - tracking branch ' origin / master ' into master - llvm - swift5 - transition
[ cmake ] Get the default value for LLVM_ENABLE_ASSERTIONS from LLVMConfig . cmake .
Merge remote - tracking branch ' origin / master ' into master - next
Bump SymbolGraph format version ( NFC )
Enhance juxtaposition binding handling to handle binary operators and functions in a more fine - grained way , allowing if / else to work without braces .
Introducing ` by ` argument label to ` Sequence . split `
The dice has been casted ^ H ^ H
AST : Remove owning addressors
Update CSDiag to use the proper InOutType convention
AST : Always diagnose request evaluator cycles
SIL : Remove special meaning for @ _semantics ( " stdlib_binary_only " )
Code completion : when completing a reference to a variable that comes from a
Sema : Rework __inout_conversion to pass inouts instead of pre - converted RawPointers .
[ Syntax ] Fix grammar and parsing for ' async ' .
Start substituting type variables through protocol conformances .
[ scale - test ] Use - stats - output - dir when available .
Merge pull request from moiseev / bitwise - ops
Merge pull request from mdiep / fix - swift - cmark - url
XFAIL Failing Debug Info tests to unblock builders . rdar : / / 20245578 .
[ test ] Temporarily weaken Reflection_KVO ' s checks due to an arm64 bug .
Merge remote - tracking branch ' origin / master ' into master - next
Do not abusing the linker for now
@ objc inference : don ' t infer @ objc through protocol conformance if the original
Remove the old " getExistentialOrArchetypeRValueAddress " hack for producing some
Merge pull request from gottesmm / pr - 92e66215cb00ac99f0a42ba2235807d017811217
Update Xcode version to Xcode 9 . 1 GM
[ NFC ] Internalize the REPL ' s LLVMContext
Merge pull request from LucianoPAlmeida / correct - test - case
remove repeated " if the " from documentation
[ Apple Silicon ] Mark several tests that require the Swift interpreter accordingly
stdlib : move Equatable , Comparable and Hashable into separate files
SimplifyCFG : Tail duplicate when coming from objc - method call blocks
Test that IUO - handling thunks use the IUO flag
When type - checking a slice type , " desugar " it to a type that is
Merge remote - tracking branch ' origin / master ' into master - next
ClangImporter : Share a module cache between CompilerInstances
Merge pull request from lanza / merge - grlp
Merge remote - tracking branch ' origin / master ' into master - rebranch
Merge remote - tracking branch ' origin / main ' into next
Merge pull request from compnerd / SwiftRemoteMirror
Merge remote - tracking branch ' origin / master ' into master - rebranch
[ stdlib ] Implement reduce with inout ( SE - 0171 )
Merge pull request from adrian - prantl / followup
[ AST Printer ] Print the " renamed " option to @ available .
Build ParamDecls ' in the REPL .
SimplifyCFG : propagateSwitchEnumCondition should not change the CFG
Merge pull request from gottesmm / emit_bind_optional_fixes
Reflection : Implement a TypeRef - > Demangle tree adapter .
Make this test actually run sometimes and pass
Merge pull request from compnerd / link - deps
Merge remote - tracking branch ' origin / master ' into master - next
Move a no - longer - crashing swift - ide - test over .
SILGen : Add AnyHashable erasure support for function conversions
Merge pull request from slavapestov / fix - invalid - method - override
Merge remote - tracking branch ' origin / master ' into master - rebranch
Runtime : Add newline to implicit ObjC entrypoint log message .
test : disable stdlib / TestData . swift due to rdar : / / 61347106
[ Serialization ] Stop using DeclID et al for " the next available ID " .
Fix the name of the initializer for initWithObjectsAndKeys .
Fix overflow issues
Merge pull request from rudkx / minor - refactoring
Merge pull request from compnerd / what - do - we - tell - failure
Small doc fix after the Float / Double typealias shuffle
SILOptimizer : fix a compile time problem in the inliner
Update WindowsBuild . md
Merge remote - tracking branch ' origin / master ' into master - next
Clean up Codable synthesis slightly
Merge pull request from gparker42 / cdtor - comment
[ ClangImporter ] Handle decls declared in imported headers .
Fix recently introduced control path warnings with Clang
Merge pull request from nkcsgexi / indent - chain - call
Referencing encoders should use parent codingPath
Merge remote - tracking branch ' origin / master ' into master - next
Increase the inliner threshold .
[ Testing ] Formalize stress tests
silence a fallthrough warning I missed .
Merge pull request from varungandhi - apple / vg - doc - new - starter - guides
AST : Fix computeSelfParam ( ) to respect __consuming on class methods
Address more suggestions to Fix thread - unsafety in Process . Argument initialization
CollectionsMoveIndices : unbreak integer ranges in the new . . < overload set
Merge pull request from slavapestov / ide - small - cleanups
Don ' t permit type definitions within protocol extensions .
stdlib : fix warnings about deprecated @ noescape and @ autoclosure syntax
Merge remote - tracking branch ' origin / master ' into master - next
Merge remote - tracking branch ' origin / master ' into master - next
[ frontend ] Added support for - enable - experimental - patterns to the integrated frontend .
[ overlay ] Change INBooleanResolutionResul . defaultProfile to . isDefaultProfile
AST : Don ' t mangle weak / unowned as part of a property ' s type
[ RLE - DSE ] . Enable redundant load elimination by default . . .
Merge pull request from smeenai / reflection
[ sil ] Add a new API ValueBase : : getSingleUser ( )
Sema : Fix inheritance of designated initializers with default arguments in generic context
[ Parse ] Add fix - it for ' rethrows ' in function type
[ ls - opts ] Make sure to invalidate StoreMap stores when we fail to eliminate a dead store .
SILGen : Handle abstraction differences in stored property access .
Merge remote - tracking branch ' origin / master ' into master - next
IRGen : Code simplification coerceValue also handles bitcast
Merge pull request from gottesmm / pr - 400a2e32a349f60ef9499e4d50c6f00b6fae5b80
SILBuilder : r / r of constant_ref is a no - op .
Make sure all non - inlining uses of SILCloner also clone the SILDebugScope
Merge pull request from rudkx / reenable - perf - hack - attempt - 2
[ Runtime demangling ] Add test for AnyObject composition case
Correct an error message
[ SourceKit ] Document source . request . cursorinfo
[ IDE ] Introduce the " syntax - coloring " action .
Register the Exclusivity benchmark .
Merge pull request from gottesmm / pr - d5b367dbe768a978c1f8a02bee26a1bffe61cfc9
[ varargs ] Add test for rdar : / / 32547102
Merge remote - tracking branch ' origin / master ' into master - next
[ Type Checker ] Remove the TypeChecker from most of the “ override ” checking .
WinSDK : add some more constants to the overlay
Merge pull request from hborla / existential - conforms - to - self - request
dispatch_source_create returns NULL but is marked up as nonnull
Move interpreter - based test selector_arguments . swift to Interpreter directory
Remove useless include as well
Merge pull request from rudkx / extend - operator - designated - type
Merge remote - tracking branch ' origin / master ' into master - next
Update for feedback from Dmitri .
[ Parse ] Don ' t try to diagnose extraneous & while parsing
Merge remote - tracking branch ' origin / master ' into master - next
Merge remote - tracking branch ' origin / master ' into master - next
Zero - initialize uninitialized Dictionary variables for the debugger .
Merge pull request from nkcsgexi / 72550007
stdlib : correct a comment to be more explicit that this is a hack
Quell warning about non - virtual destructor .
[ Constraint generation ] Collect all closure body references to type variables .
[ IDE ] [ Index ] Fix syntax coloring , index , and rename support for custom attributes
[ ABI ] Emit separate symbols for protocol conformance descriptors .
[ NFC ] Fix comment typo in build - script - impl
Merge remote - tracking branch ' origin / master ' into master - next
IRGen for varargs tuples .
[ CMake ] Remove ' check - swift ' dependency on internal tool .
Add . skip to non - core tests
Runtime : Let structural types be cast out of boxed AnyObjects too .
Merge pull request from atrick / cmake - stdlib - exclusivity
test : add a linker directive test for ios
Add a bit and some diagnostics logic to help with the ongoing id - as - any project .
ClangImporter : Allow unions and structs with bitfields to be imported .
Merge pull request from nkcsgexi / 64373983
Merge pull request from ravikandhadai / postdom - utility
Merge remote - tracking branch ' origin / master ' into master - next
Run stable ABI tests on all Apple platforms
Merge pull request from erg / python - fixes2
Merge pull request from compnerd / off - with - your - head
Use ParserResult and ParserStatus in tuple and function argument parsing
CopyForward : reset postorder traversal each iteration .
Add explicit init from BinaryInteger to CGFloat
[ test ] Add missing REQUIRES .
[ IRGen ] Fix SynthesizedFileUnit . ( )
Fix typo in RedundantLoadElimination . cpp
Merge pull request from bob - wilson / master - next - fixes
[ ownership ] Move BorrowScopeOperand : : visitEndScopeInstructions ( ) from . h into . cpp .
Sema : Fix crash when coercing a closure literal returning one uninhabited type to another
migrator : correct a misplaced assertion . rdar : / / 31980368
Disable more failing tests on iOS
Merge remote - tracking branch ' origin / master ' into master - next
Merge remote - tracking branch ' origin / master ' into master - next
Merge remote - tracking branch ' origin / master ' into master - next
AST : Removed extraneous space from let / var as a parameter fixit .
Add a change I forgot to add in the previous commit
[ GenericSignatureBuilder ] Consume where clauses on associated types .
Merge pull request from compnerd / winsk
Merge remote - tracking branch ' origin / master ' into master - next
[ Test ] Disable Prototypes / BigInt . swift .
[ test ] Collection tests with reference types should also require optimized stdlib
[ Diagnostics ] - suppress - warnings and - warnings - as - errors flags
Merge pull request from CodaFi / just - my - type
[ CMake ] Cleanup Swift library tablegen dependencies
Progress towards generic types .
Commit updated ' Failable Initializers ' proposal .
[ Trailing closures ] Attempt both forward and backward scans .
fix typo : commiting - > committing
[ ABI ] Add associated conformance descriptors .
Merge pull request from ahoppen / backing - type - of - DeclName
[ docs ] Add First Pull Request guide and Getting Started guide .
XFAIL test for optimized swift_test_modes
Merge pull request from compnerd / iphoneos - sdk
[ Serialization ] Simplify reading of group info for swiftdoc files ( )
Merge remote - tracking branch ' origin / master ' into master - next
[ Gardening ] Redocument TypeExpr ' s Implicit Constructors
Revert " Use pthreads in the metadata runtime unit test . "
[ SourceKit ] Add missing includes
Merge pull request from bitjammer / acgarland / rdar - 61746582 - dont - use - innermost - context
[ Typechecker ] Synthesize _modify for storage wrapper correctly
[ ObjcHeaderPrint ] When initializing CompilerInvocation with clang tuples , update i386 to x86_64 so that Stdlib is available . rdar : / / 22978765
Grammar and formatting
SILParser : small refactoring .
Debug Info : Forge a name getters and setters using their parent declaration .
Merge remote - tracking branch ' origin / master ' into master - next
Merge pull request from DougGregor / canonicalize - conformance - access - path
NoReturn folding : Set unreachable uses in different basic blocks to undef
merge more and harder .
[ stdlib ] Remove a redundant and un - specific protocol extension
Fixing expr return .
Make the testcase target independent .
Attempt to fix DebugInfo / self . swift on 32 - bit
Merge pull request from apple / on - the - rpath - to - greatness
Add CHANGELOG note about long double < - > Float80 ( )
Pass flag for addGenericParameters
Push a generic root class through the interpreter .
Merge pull request from natecook1000 / natecook - utf16 - randomaccess
Merge pull request from modocache / test - irgen - report - dead - method - call - stdlibunittest
Merge pull request from xedin / DSP - 1794
Merge pull request from moiseev / reverse - improvements
[ sil - aa ] Make sure we do not use too much compile time in valueMayBeCaptured by limiting the size of the use lists we will process to 32 .
Merge pull request from slavapestov / more - metadata - layout - usage
Revert " Merge pull request from rintaro / syntaxparse - exprcollection "
Fix typo
Merge pull request from slavapestov / fix - constructor - lookup - circularity
[ test ] Add test for deserialization of generic protocol witnesses ( )
[ TypeChecker . rst ] NFC : Mark that ` @ dynamicCallable ` diagnostics have been ported
Reword and clarify an awful paragraph .
IRGen : Fix implementation of emitValueCaseTest for multi - payload enums .
Handle ` nil ` with any number of parentheses
Merge remote - tracking branch ' origin / main ' into next
SIL : Add an " UnownedInnerPointer " result convention .
Update stdlib / private / SwiftPrivate / SwiftPrivate . swift
AST : Fix crash in getContextSubstitutions ( ) when a class has a malformed superclass type
random typo
SIL : Add BridgeToBlockInst .
[ Docs ] Remove FrontendStatsTracer task from request - evaluator open projects .
[ benchmark ] Rename SampleRunner - > TestRunner
[ Docs ] [ AutoDiff ] Correct AD builtins .
Load cross - import overlays
[ Serialization ] Clarify role of matchAttributes in FileUnit
[ TypeChecker ] NFC : Add one more test - case for SR - 11743
Merge pull request from benlangmuir / pkg - build - from - tc - combined
More watchOS test churn
Merge remote - tracking branch ' origin / master ' into master - next
recent diagnostics changes fixed 5 crashers and broke 1 .
Merge remote - tracking branch ' origin / master ' into master - next
[ Parser ] Make PersistentParserState parameter optional and remove it from the constructor that accepts a StringRef fragment .
Merge remote - tracking branch ' origin / master ' into master - next
Merge pull request from shajrawi / address_enum
Merge pull request from DougGregor / repeal - conformance - death - tax
[ stdlib ] Add other comparison operators to 0 - ary tuples
[ test ] Make sure member completion recognizes useful ' @ ' commands .
stdlib : add an API AnySequence . init ( ( ) - > Generator )
Merge remote - tracking branch ' origin / master ' into master - next
Merge remote - tracking branch ' origin / master ' into master - next
rework and tidy SILGenLValue : : visitMemberRefExpr to make the logic
Merge pull request from ahoppen / fix - compile - error
Towards s390x support ( )
Implement the first half of < rdar : / / problem / 15922884 > support non - member didset / willset properties
[ AutoDiff upstream ] Test differentiable collection higher - order functions .
Merge pull request from DougGregor / module - free - gsb
[ CSDiag ] We have trailing closure info for subscripts now . Use it .
Merge remote - tracking branch ' origin / master ' into master - next
Move cache check into TypeRefs ' static create methods
Merge pull request from jrose - apple / deprecation - and - depreciation - are - two - different - words
Merge pull request from dainmiller / patch - 1
When not used for an escaped identifier , back - tick is a token .
Omit needless words : distinguish class vs . instance properties for pruning .
[ Tests ] Consolidate stress / long tests in validation - tests
[ PrintAsObjC ] Don ' t forward - declare @ objc enums using typedefs .
stdlib : remove Word and UWord
MetadataReader infrastructure for reading capture descriptors
Don ' t add implicit dynamic to variables to local and implicit vars
SILGen : Fix override handling when a dynamic init is also required .
Merge pull request from nkcsgexi / syntax - parser - opt
remove some dead protocol logic .
Add the value witness marker for generic builtins .
[ Sema ] Synthesize Eq / Hash for enums with payloads , structs
Merge remote - tracking branch ' origin / master ' into master - next
[ Frontend ] Disable skipping any function bodies for SwiftOnoneSupport
[ benchmark ] Premultiply N for case conversion benchmark
Merge pull request from varungandhi - apple / vg - fix - module - trace - cycle
Merge pull request from apple / sr - 9348
Merge pull request from MForster / import - const - members
Sema : Fix logic inversion that threw away significant bits of enum raw value literals .
[ AST ] Rename EmittedMembersRequest - > SemanticMembersRequest .
Remove retain / retain_autoreleased / release / retain_unowned from sil - mode .
Fixes requested by Jordan when reviewing 389238e801db7b0c027c473787caf0898d37151d .
Enable mandatory inlining of generic functions with unbound generic types .
Merge remote - tracking branch ' origin / master ' into master - next
Merge remote - tracking branch ' origin / master ' into master - next
Merge pull request from nkcsgexi / property - bang
Allow OperatorLookupDescriptor to hold a file or module
[ sourcekitd ] [ CursorInfo ] Make sure we handle the implict VarDelcs being introduced in CastStmt bodies
Steal IRGen TypeVisitor for SILGen .
[ stdlib ] Get NewArray test building and passing
Ignore missing types on for statement expressions
Temporarily soft - fail this assertion .
Merge remote - tracking branch ' origin / master ' into master - next
Kill a stray TAB
Tolerate non - empty braces in braces
Decouple the solution to a constraint system from the constraint system itself .
[ test / TBD ] Input - > Inputs . Whoops .
Revert " stdlib : change sort ( ) and sorted ( ) into methods "
Replace " oneof " with " union " . . . everywhere .
Handle same - type constraints in GenericFunctionType : : partialSubstGenericArgs .
[ Parse ] Migrate tests to Swift4
Evolution : Compare symbols before and after
[ ClangImporter ] Don ' t put forward - declared structs in the lookup table .
Merge remote - tracking branch ' origin / master ' into master - rebranch
[ CSGen ] Replace ` getInterfaceType ( ) - > mapTypeIntoContext ( ) ` for VarDecl with ` getType `
rework how let - ness of ' self ' is determined , to be based on the type of self .
[ IRGen ] Don ' t add prespecialized metadata to ABI .
Merge pull request from AnthonyLatsis / selfrefkind - refactor
[ Frontend ] Move initial REPL file creation logic
Merge pull request from slavapestov / inout - closure - regression - test
Simplify some code , NFC .
Merge pull request from atrick / fix - assert - termoper
Merge pull request from Gankro / dsindex
[ stdlib ] Quick fix
my fix fixed this testcase too .
[ AST ] Optimize lazy - member storage for nominal type and extension declarations .
Fixed crasher 28707
Factor the resolution of witnesses via derivation into its own routine .
[ Parse ] [ Swift3 ] Add a warning for unsupported conditional compiliation expression
Merge pull request from CodaFi / type - check
[ string ] Un - break Linux build
Merge pull request from atrick / fix - temprvalue - destroy
[ ImportName ] Static - ize many more methods
Remove the case ' Invalid ' from the TermKind enum and just use an unreachable in the ValueKind - > TermKind switch instead .
Use # selector in StdlibUnitTest ' s Foundation extras
Record dependency for cached results .
SIL : Introduce a SILModule object .
ditionary - decodable - fix
Merge remote - tracking branch ' origin / master ' into master - next
15242776 docs : Replace " Slice " with " Array "
Fix build with SWIFT_RUNTIME_MACHO_NO_DYLD = 1 in ImageInspectionStatic . cpp
Remove dead function .
Merge pull request from compnerd / explicit - BlocksRuntime
Merge pull request from slavapestov / c - self - capture - crash
Fix Constraints / diagnostics . swift on arm targets ( No Float80 type ) .
Runtime : Fix a warning
Add reference to a rdar for TODO in comments .
[ benchmark ] Disable inlining of wCSIA in benchmark
spacing fix .
[ swift - reflection - dump ] Support for ELF32 .
Add a ` MemoryLayout < T > . offset ( of : ) ` method for getting the offset of inline storage .
Merge remote - tracking branch ' origin / master ' into master - next
compiles , unfmt
IRGen : Remove stale FIXME .
[ semantic - sil ] Implement ValueOwnershipKindBuiltinVisitor to compute ValueOwnershipKind for Builtins / llvm intrinsics .
[ Driver / Frontend ] Remove the ' - module - cache - path = ' alias .
Merge pull request from apple / emacs - mode - parse - where - clause
[ CSDiagnostics ] Add invalid partial apply diagnostic for mutating methods
[ AST ] NFC : Attr . def clean up ( )
Substantially rework how weak pointers are represented in the AST and
[ SourceKit ] Disable module system headers validation
Merge remote - tracking branch ' origin / master ' into master - next
Use code style in documentation for referred functions .
Improve some incorrect label diagnostics , and pull the logic for it into CSDiag ' s core .
Update ClangImporter and IRGenDebugInfo to build with upstream clang
[ test ] update test case
Merge remote - tracking branch ' origin / master ' into master - next
Add a Mirror for Interval types
Implement support for calling imported C functions from Swift .
Merge pull request from Rostepher / lay - me - out - to - dry
Merge pull request from compnerd / nullability - portability
Move some closure related testcases out to Constraints / closures . swift ,
` appendContentsOf ` = > ` appendContents ( of : ) `
[ CodeCompletion ] Infer keypath root type from the context type
[ gardening ] Use sys . exit instead of exit
Merge pull request from bob - wilson / reenable - test
Merge pull request from phausler / swift - 3 . 1 - branch
SIL : get IntLiteral / FloatLiteral values from AST
Clang importer : centralize the " suppress declaration import " logic .
Merge remote - tracking branch ' origin / master ' into master - next
Revert " [ ownership ] Ignore non - consuming uses in dead end blocks when we are analyzing objects that are consumed once in the same block in which they are defined . "
[ AccessEnforcementOpts ] Add merge accesses analysis to conflict analysis
Merge pull request from aschwaighofer / make_prespecialization_experimental
Dictionary : More guaranteedNative fast paths
Specify linker for Android armv7 target .
ClangImporter : Mark generated union accessors as transparent
Merge remote - tracking branch ' origin / master ' into master - llvm - swift5 - transition
Merge remote - tracking branch ' origin / master ' into master - next
Support for stack promotion in IRGen .
Recent fixes fixed this testcase as well .
Add tests for frontend and driver .
Merge pull request from xedin / mark - 26564101 - as - fast
remove " RequiresAnotherPass " from PreCheckExpression , it is always false .
tests : remove the now unused option - new - mangling - for - tests
Merge pull request from gregomni / generic - func - args
Add basic CString < - > UTF8 API variants .
vim : highlight the ` throws ` modifier
Fix typos
Sema : Fix for circularity in unqualified lookup when resolving DeclRefExpr
use claim for trailing closure
Merge pull request from rintaro / syntaxparse - genericparam
Really fix this test by creating separate tests one for when the stdlib is compiled with checks and one where it is not .
Merge remote - tracking branch ' origin / master ' into master - next
Merge pull request from practicalswift / typo - fixes - 20160927
[ SIL ] NFC : Repack misc StoreReferenceInstBase bits
Downgrade type access checks to warnings in certain cases .
Rename PostOrderInfo = > PostOrderFunctionInfo to match other FunctionAnalysisBase subclass ' s function info .
[ gardening ] Fix incorrect spacing .
Merge remote - tracking branch ' origin / master ' into master - next
[ SIL ] NFC : Repack SILBoxTypeRepr bits via LLVM data types
Merge pull request from gottesmm / pr - 339c783b678f40183e3a6a3b4df5d136ad82f035
Merge remote - tracking branch ' origin / master ' into master - next
Merge pull request from DougGregor / record - conformance - in - source - file
[ Doc ] Add some notes about the recently added doc comment fields .
[ test ] Update libgcc path for Android NDK r12
[ benchmark ] Round quantile idx to nearest or even
Attemp to fix python lint errors
do an inttoptr / ptrtoint dance to enable atomics on Builtin . RawPointer ,
Merge pull request from porglezomp - misc / fix / stdlib - unittest - over - release
[ CMake , build - script ] Add support for cross - compiling tools .
tests : relax check commands in IRGen / struct_resilience . swift
SIL : Clean up SILType interface for functions .
[ projection ] Add a new data structure for use in NewProjection called PointerIntEnum .
Merge remote - tracking branch ' origin / master ' into master - next
Merge remote - tracking branch ' origin / master ' into master - next
Wrap a few functions from LLVM SourceMgr in preparation of making
Teach escape analysis about the semantics of copy_addr instructions
master - next : Adjust for LLVM r328165 moving a header file
RLE
Make protocol thunks expect a metatype pointer ( for This ) as
Update new tests for linux .
Merge remote - tracking branch ' origin / master ' into master - next
Test case
Per API review , rename ' reserve ' to ' reserveCapacity ' .
Make ' T . self is U . Type ' work .
Merge pull request from trentxintong / MISC
Merge pull request from CodaFi / array - of - sunshine
Clear out the " escaped identifier " bit when we form a token .
[ GSB ] Don ' t eagerly minimize the term - rewriting system .
docs : add some docs covering Windows cross - compiling
Merge pull request from compnerd / foundation - framework
build : workaround incorrectly cased import libraries in WinSDK
[ Test ] Add a test to make sure every group has decls to print .
Merge remote - tracking branch ' origin / master ' into master - next
Merge pull request from amraboelela / android - toolchain
Sema : Remove ArgumentTupleConversion constraint now that its no longer used
Merge pull request from atrick / fix
Eliminate race in swift_bridgeErrorToNSError .
Sema : Fix requirement / witness disambiguation for subclass existentials
Merge pull request from apple / add - jobs - flag - build - script
Add test missing from r4311 .
IRGen / Runtime : Populate the runtime protocol conformance table with relative references .
IRGen : Clean up class archetype spare bits hack
Merge pull request from hamishknight / better - late - than - never
[ Docs ] Fix minor code typo in SILPro . . Man . . md
build : do not consult the legacy location for libraries
Update README . md ( )
Don ' t jumpthread through loop headers
[ sink ] Fix a testcase that the verifier rejected .
Remove stale FIXME .
[ SE - 0095 ] [ Runtime ] , [ Demangler ] , & AST printer updated to new composition syntax
[ stdlib ] Include Float16 among source types recognized for generic conversion
Mark internal inits with underscored parameter .
Don ' t error out when the fast ABI is unavailable
Diagnose clobbering clang ' s VFS with our own
Merge pull request from owenv / profiler - fix
Add Set type .
[ sil - parser ] Call default Substitution Constructor so we initialize an ArrayRef < Conformance > correctly .
Merge remote - tracking branch ' origin / master ' into master - next
IRGen : Use @ _weakLinked to test backward deployment of resilient protocols
Introduce C parser library
Merge remote - tracking branch ' origin / master ' into master - next
Rename generic arguments
IRGen : Coerce parameters using the larger of the from and to types .
Merge remote - tracking branch ' origin / master ' into master - llvm - swift5 - transition
[ gardening ] Fix a vs an typo
[ stdlib ] Silence a few assert related warnings
[ sil ] Teach StructInsts how to return the operand for a given VarDecl .
Speed up roundtrip syntax tests
Merge remote - tracking branch ' origin / master ' into master - next
Update a test case ' s expected output
IUO : Generate Optional < T > rather than ImplicitlyUnwrappedOptional < T > .
Merge remote - tracking branch ' origin / master ' into master - next
SIL : Rename SILFunction : : getMangledName to SILFunction : : getName .
Use StringLiteralExpr node if it ' s a single literal segment
[ Runtime ] Return nullptr instead of NULL in lookupSection stubs .
Update stdlib / public / core / Indices . swift
Fix Evolution tests in the mode when StdlibUnittest is built with - sil - serialize - all
[ ConstraintSystem ] Replace ` Fix ` with ` ConstraintFix ` throughout solver
[ Testing QoI ] Make the Windows link job test more robust
Clang importer : rename importDeclContext - > importDeclContextImpl to emphasize
AST : Remove an overload of GenericSignature : : getSubstitutions ( )
Add an XFAILed KVO - with - dynamic test as a reverse - canary for < rdar : / / problem / 17697837 > .
CMake : Skeletal infrastructure for building multiple variants of Swift libraries ( for iOS , iOS simulator , etc . ) .
Merge remote - tracking branch ' origin / master ' into master - next
Merge pull request from practicalswift / remove - unused - variable
Fix a compiler_crasher test to not depend on specific diagnostics
Move Utilities back to TypeChecker
Simplifying the flow ( )
[ benchmarks ] Rename column label SPEEDUP to RATIO
AST : Compress the mangling for generic signatures a bit .
Merge remote - tracking branch ' origin / master ' into master - rebranch
Merge remote - tracking branch ' origin / master ' into master - next
Corrected typo : separatly - > separately
Merge pull request from glessard / always - never - testpredicate
[ stdlib ] [ swift - 3 - indexing - model ] removing Strideable requirement for checkBidirectionalCollection
tests : make ` jobstats ` and ` process - stats - dir ` python3 compatible
[ tools ] gyb : Add two missing Python " dedent keywords "
Add a test case .
build - script : handle special build rules for non - atomic runtime
Revert " [ Apple Silicon ] Adjust linking deployment target from 11 . 0 down to 10 . 16 on x86 . "
Merge pull request from LucianoPAlmeida / validation - test - match - generic - signature
Added a FIXME and removed a get per review suggestions .
Merge pull request from practicalswift / sourcekit - 008 - swift - typechecker - typecheckfunctionbodyuntil
Remove inaccurate comment .
AST : Don ' t call hasType ( ) / getType ( ) / setType ( ) on AbstractTypeParamDecls
Don ' t treat missing imports as a fatal error in REPL mode .
[ const evaluator ] Parameterize allocation of symbolic values in the
[ Parse / Syntax ] Refactoring to decouple the parser from syntax tree creation
Merge remote - tracking branch ' origin / master ' into master - next
[ android ] Fix function signature to match usage in libdispatch .
Merge pull request from apple / SR - 5401
Adjust for SVN r297121
[ Parser ] When delay - parsing a function body , make sure the closing brace is included .
Merge remote - tracking branch ' origin / master ' into master - rebranch
Choose correct IR types for @ objc return values .
Merge remote - tracking branch ' origin / master ' into master - next
[ build - script ] Merged the PlaygroundLogger and PlaygroundSupport products .
Merge pull request from practicalswift / bridging_objcbridgeable_missing_conformance
Add FIXMEs to code completion code .
Merge pull request from xedin / perf - 33935430
llvm - as is used by a few tests ( e . g . Frontend / embed - bitcode . ll ) , so list it
Merge pull request from apple / string - improvements
[ serialization ] Push record layout info into a private header .
[ parse ] Recover better from malformed subscript decls for code - completion
Merge pull request from adrian - prantl / 43566087
Teach IRGen ' s archetype uniquing to handle same - type constraints more generally .
Merge pull request from eeckstein / fix - cowarrayopt
[ build ] Refer to clang path via LLVM_ { SRC , OBJ } _ROOT .
[ ConstraintSystem ] Refactor ` recordFix ` to no assume that non - augmenting fixes anchored on expressions
[ libIDE ] Add syntax kind of argument to fix a crash . rdar : / / 27377118
SILGen : Implement r / r of fragile aggregates .
Merge pull request from compnerd / linker
Merge pull request from ikesyo / os - macos
Release note ' Generator ' protocol .
TempRValueElimination : handle potential modifications of the copy - source in a called functions correctly .
Merge remote - tracking branch ' origin / master ' into master - next
test : Fix IRGen / c_layout . sil to work on powerpc64 { le }
Merge remote - tracking branch ' origin / master ' into master - next
Remove some of the passes at the late stages of the optimization pipeline .
Merge remote - tracking branch ' origin / master ' into master - next
Merge pull request from gottesmm / pr - 1639094f806f1130063c975fee6b1f74f2babb6a
Merge pull request from gottesmm / pr - 12238ae506fe68efdd321c0ed03cfc730d340d2c
[ lower - aggregate - instrs ] Add an additional type lowering style " DeepNoEnum " for use in lowering aggregate memory operations .
Merge remote - tracking branch ' origin / master ' into master - rebranch
[ swiftc ( 38 vs . 5569 ) ] Add crasher in swift : : TypeBase : : getContextSubstitutions
master - next : temporarily disable a failing test
[ AutoDiff ] Fix differentiation transform crashers in library evolution mode . ( )
. . . and get rid of a bunch of dashes .
Add comments .
Gardening : Fix some unused variable warnings in no - assert builds
Change ` element ` to ` result `
[ Build System : Python ] Cleaned - up the python_lint . py script .
Comment fix .
Sema : Allow local types to conform to protocols with associated types
SIL : rename getCalledFunction to getCalleeFunction in ApplyInstBase
Rename - simplify - cfg to - jumpthread - simplify - cfg .
Runtime : add a workaround for Windows ARM64
Merge remote - tracking branch ' origin / master ' into master - next
[ AsyncCC ] Resolve metadata from class instances .
Just reuse old body when reverting return insertion .
Introduce a conversion - to - supertype expression to capture conversions
RemoteAST : Fix Error existential introspection on Linux
Add build - script option - - skip - test - toolchain - benchmarks
[ NamedLazyMemberLoading ] Flip flag polarity : on by default , optionally off .
[ sil - devirtualizer ] Always try to devirtualize the invocation of a method from a static class of an instance , even if the default case cannot be devirtualized completely .
Improvements to the API of struct URL .
[ Stats ] Fix typo .
Merge remote - tracking branch ' origin / master ' into master - next
Merge pull request from rudkx / cs - typemap
stdlib : remove pthread_barrier attributes
[ ClangImporter ] Use the current version to vend enum constant names .
If we do not move the retain / release instruction , we do not need to
Refactor OSX archs for job config
Merge remote - tracking branch ' origin / master ' into master - next
Pack cleanups a little better .
[ stdlib ] Excise unsafeBitCasts from SliceBuffer
Add doc whitespace to fix Sphinx error .
Canonize swift header files headers and footers
Add expectCrash function to StdlibUnittest
adjust testsuite to put @ autoclosure on decls instead of types , and remove a
[ SourceKit ] Fix flakey completion tests
Merge pull request from rintaro / ide - completion - rdar59817674
migrator : use the right method to get the location of return type . rdar : / / 32545812
Merge pull request from milseman / whoops
Change ' x . type ' to ' x . dynamicType ' .
AST : Print AnyObject layout constraint as ' : AnyObject ' and not ' : _Class '
[ QoI ] Cleanup AST after trying to shrink constraint system of invalid expression
Merge remote - tracking branch ' origin / master ' into master - next
change this to isa < > since we ' re using a bool return .
[ stdlib ] moving String . CharacterView . Index . _predecessor / _successor to index ( before : ) / index ( after : )
Merge pull request from gottesmm / pr - f84811812599fd979ceedad733e34a66ca2d6d9d
Resolve a fixed crasher
AST : Remove a few usages of TypeBase : : getInOutObjectType ( )
stdlib : explain the trick used to achieve @ noinline without @ noinline
Clean up some trivially true condition clauses
Add code to CopyAddr for replacing its Src / Dest arguments .
[ dtrace ] Add dtrace for support for counting the number of retains / releases applied to specific classes as well as computing total retain / release counts .
[ IDE ] Handle observing accessors for syntax and semantic annotations .
[ benchmark ] Refactor run runBenchmarks logVerbose
Merge remote - tracking branch ' origin / master ' into master - next
Fixing spacing .
[ FixCode ] When fixing missing enum elements in a switch statement , print associated payloads as well . ( )
Merge pull request from slavapestov / default - witness - thunks - multithreaded
[ swiftc ( 137 vs . 5186 ) ] Add crasher in swift : : ASTVisitor
Merge pull request from eeckstein / fix - stack - nesting
SILOptimizer : handle copies in LetPropertiesOpt .
Fix assertion failure completing anonymous default arguments
Cleanup actor protocol conformance error
Revert " [ Type checker ] Recurse into statements when cleaning type variables , harder . "
release note the inout syntax change .
[ GSB ] Infer same - type constraints among same - named associated types .
Perform collection force - casts by force - casting the elements
Revert " [ stdlib ] Speculative workaround for ASAN failure "
Add a test case for global dead store elimination , making sure it does not remove stores
Merge pull request from gottesmm / pr - a3e66535385c13130292d0422b5be34cad234a13
Sema : Update getDefaultGenericArgumentsString ( ) for primitive AnyObject
Merge pull request from guitard0g / sp_entrypoint
fix a pedantic warning .
Merge pull request from frootloops / else
Merge pull request from graydon / compiler - performance - doc
Merge remote - tracking branch ' origin / master ' into master - next
[ AutoDiff ] NFC : prettify commutative diagrams . ( )
Add rudimentary nested closure param / return / throws support for Doxygen
Fix exit bugs of some python scripts
Follow the base type if the renamed attribute is point to another type
Merge remote - tracking branch ' origin / master ' into master - next
Merge pull request from eeckstein / fix_aeic
Debug Info : Handle vector types when extending variable live ranges .
stdlib : Fix for NSFastEnumerator breakage in resilient builds
[ RemoteMirrors ] Fix the pointer - returning readBytes wrapper to actually work .
EscapeAnalysis : minor cleanup to prepare for alias analysis rewrite .
switch a bunch of instructions ( e . g . load ) to print types for their operand .
IRGen : Give ObjC metadata symbols internal linkage .
Condense benchmarks
add requiresReabstraction flag / check ( s ) to RValueEmission
[ test ] [ gardening ] move test files to appropriate directory
Merge pull request from jckarter / nsmutabledictionary - overlay - subscript
Teach the walker to walk inside ExtensionDecls correctly .
As soon as a ( partial ) solution becomes worse than the best solution , stop .
Remove LLVM_INITIALIZE from swift - demangle .
Sema : Coerce structural lvalues when binding to non - lvalue type variables .
SIL : Non - final public methods of resilient classes don ' t need public linkage
build : remove set of internal CMake variable
Merge pull request from jrose - apple / it - is - a - lot - to - process
Merge remote - tracking branch ' origin / master ' into master - next
Merge pull request from jckarter / trimming - metadata
Merge pull request from compnerd / compiler - switcheroo - too
New indexing model : fix test / SILOptimizer / prespecialize . swift
Merge pull request from nkcsgexi / forgive - no - explicit - module - map
Remove the T ( x ) - as - coercion syntax from the type checker .
Add another StringSort CPP reference .
Merge pull request from adrian - prantl / 28859432
Fix property wrappers with tuples .
[ IDE ] Don ' t assume a platform name ' s SourceRange matches its length .
EscapeAnalysis : fix problem of missing points - to edge in the connection graph .
correctly reject @ final on global functions instead of crashing .
Add a partial dead store test case
Merge remote - tracking branch ' origin / master ' into master - next
Type - check both sides of a PatternBindingDecl within the same constraint
Merge pull request from mikeash / rename - avoid - more - objc - conflicts
Strip TypeLoc from IsPattern
Switch a few more clients over to Type : : findIf ( ) .
Switch the generic heap box metadata cache over to use ConcurrentMap directly .
[ SE - 0206 ] [ stdlib ] De - underscore Hashable . hash ( into : )
Adapt to upstream header file reshuffling
Several NFC changes :
SIL . rst : Tidy up existential container description
SILCombine : Add another index_raw_pointer - > index_addr peephole
Add SILGenSourceFileRequest
Merge remote - tracking branch ' origin / master ' into master - next
[ SE - 0134 ] Another fixup for UTF8 - related property renaming
[ Driver ] Provide a single API to iterate over all valid file types .
fix < rdar : / / problem / 21679169 > compiler crashes on " { ( = _ = _ ) in "
Merge pull request from owenv / pretty - printed - diags - source - location - fixes
[ Foundation ] Add initializers for NSRange < - - > Range ( )
Change the layout of opaque existentials so that
Merge pull request from drodriguez / android - enable - some - more - tests
[ concurrency ] SIL : fix operand ownership of hop_to_executor
Use the archetype builder ’ s type resolution more directly .
ExplicitModule : remove all search paths for command lines to explicitly build Swift textual interface
implement parsing , AST , and basic Sema support for ' defer ' .
Merge pull request from rudkx / inline - asm - when - optimizing
Merge pull request from slavapestov / dynamic - modify - regression
This test change was a mistake ; I didn ' t realize that there were API notes in play .
Merge pull request from eeckstein / stringswitch - abi
Merge pull request from rjmccall / sil - coroutines
thread a vfs through the completion handler
convert the stdlib to use the new inout syntax .
ABI / API checker : check if module has failed to load and abort if so
[ gardening ] Remove dupe word
Merge remote - tracking branch ' origin / master ' into master - next
SILGen : ArgEmitter : : forward ( ) should return PreparedArguments and not an ArgumentSource
[ RLE - DSE ] Refactor in RLE , move declarations around . NFC
[ Omit needless words ] " de - " is a verb prefix .
[ SE - 0095 Migration ] Migrate protocol < . . . > usage
[ benchmark ] Legacy factor ArraySetElement
[ CMake ] Remove accidentally - added extra quotes from SWIFT_COMPILER_VERSION .
[ stdlib ] Refactor min ( _ : _ : _ : _ : )
[ stdlib ] Rename Slice = > ArraySlice
use isa instead of dyncast in a boolean context , NFC .
Merge pull request from gottesmm / do_not_compute_directly_paths_to_llvmlibraries_use_cmake_instead
[ CodeCompletion ] Don ' t show editor placeholders as completions
Merge pull request from jrose - apple / translation - reflection - but - what - about - rotation
[ Mangling ] Add a new mangling for opaque return type to use when mangling an ObjC runtime name ( )
Merge pull request from hartbit / fix - crasher
Merge pull request from benlangmuir / sourcekitd - test - enhancements
Merge pull request from landonf / landonf / freebsd - patchset - 1
[ Archetype builder ] Check concrete types against superclass constraints .
benchmarks : add a benchmark for string enum construction from a string raw value .
Merge pull request from practicalswift / sourcekit - 105 - swift - derivedconformance - deriverawrepresentable
Merge remote - tracking branch ' origin / master ' into master - next
[ stubs ] Add nullability annotations ; NFC
[ swiftc ( 55 vs . 5458 ) ] Add crasher in swift : : TypeChecker : : resolveTypeInContext
[ Test ] Start to add SILGen tests for local property wrappers .
Merge remote - tracking branch ' origin / main ' into next
[ gardening ] Update some code to use more modern SILGenBuilder APIs .
[ ClangImporter ] Import ' NSObject < FooProto > * ' as ' id < NSObject , FooProto > ' .
Merge pull request from hborla / type - cannot - conform - diag
Merge pull request from tjw / SR - 1226
arm64e does not back - deploy
Add missing comment after . h file # endif
Properly type " a as T [ ] " as a Bool , not an i1 .
[ gardening ] Fix incorrect spacing
Merge remote - tracking branch ' origin / master ' into master - next
[ g - arc - opts ] Look through non phi rc identity preserving arguments .
Give ASTContext TypeCheckerOptions
Move merging of CapturedValue flags into the class itself so that it ' ll be more clear that dealing with this will be required if more flags are added in the future . And mergeFlags correctly on the dynamic self capture .
Switch up @ objc enum printing
Allow Selector and ObjCBool in @ objc methods .
Provide shims for fork / execv for watchOS and tvOS
Merge pull request from bnbarham / benb / allow - errors - 69815975
Merge pull request from practicalswift / re - add - 28615
Sema : Split off WitnessChecker from ConformanceChecker , NFC
Code completion : complete in type contexts - - basic implementation
[ NFC ] Document the new type variable merging heuristic in addJoinConstraint
Frontend : add ` - autolink - library = ` option to support Windows
Assorted fixes to the emission of materializeForSet accessors ,
[ serialization ] Check cyclic dependencies for typealiases better .
[ CodeComplete ] Show # available in the list of keywords . rdar : / / 23039417
Merge pull request from xedin / remove - label - mismatch - from - lookup
[ test ] Factor out a module just for objc_nonnull_lie_hack . swift .
Merge pull request from artemcm / dependencies - prescan
IDE : Fix mismatched ErrorType check
Teach the type - checker to bind and capture optional
[ SourceKit ] Avoid the ' error : no input files ' diagnostic notification for syntactic - only requests
Merge pull request from gwynne / sr - 12842 - range - pattern - match - op - missing - docs
Append newline to printout of " Defaulted constraints "
[ Driver ] Add the notion of " extra environment variables " to Jobs .
[ CMake ] Include ' Remote ' headers in Xcode .
[ CS ] Resolve callees for key path components ( )
[ SE - 0206 ] [ stdlib ] Make Hasher . finalize ( ) nonmutating but __consuming
stdlib / String : fix hasPrefix ( ) and hasSuffix ( ) to perform proper Unicode
[ specializer ] Teach the specializer how to specialize partial apply insts .
Merge remote - tracking branch ' origin / master ' into master - next
Ignore missing foundation and libdispatch build directories when attempting to remove them on non - darwin platforms .
Remove ScalarToTupleExpr in favor of a flag on TupleShuffleExpr .
stdlib : install the image registrar in the static runtime
Set the kill bit for the store at the end of the basic block where the stored location is de - allocated .
[ build - preset ] Skip building benchmarks on the XCTest development preset
[ stdlib ] Enable Mirror tests on non - Darwin platforms
whitelist missing enum cases fixits .
stdlib : reattach the comment to the declaration
Excise " Accessibility " from the compiler ( 2 / 3 )
[ Coverage ] Simplify a check ( NFC )
Remove TypeChecker : : Diags
Add tests for non - single - expression closures and remove dead code
Merge remote - tracking branch ' origin / master ' into master - next
[ sil ] Extract out ApplySite / FullApplySite into their own header .
AST Verifier : add a check that AssociatedTypeDecl has a protocol DeclContext
Remove unnecessary platform - specific checks from tests
Migration to SwiftpM ’ s new boostrap script
[ Serialization ] Finish factoring the switch out to DeclSerializer
Merge pull request from jckarter / conformance - lookup - callbacks
IDE : IWYU ( NFC )
Minor formatting changes to conform to conventions .
Expand out radix detection in applyTypeToLiteral so it uses the same rules as the lexer , instead of using StringRef ' s autosensing rules . This makes octal literals work correctly . Finishes < rdar : / / problem / 10700832 > .
Add a performance unit test for stack promotion
[ stdlib ] Delete an obsolete prototype
Make test 32bit safe
[ ModuleInterfaces ] ParseableInterfaceBuilder - > ModuleInterfaceBuilder
Merge remote - tracking branch ' origin / master ' into master - next
[ stdlibunittest ] WIP uniformity / de - boilerplating
Merge remote - tracking branch ' origin / master ' into master - llvm - swift5 - transition
Add more unicode characters to one test , and add a new test for escaping them in the supplementary output map .
Merge remote - tracking branch ' origin / master ' into master - next
Full IRGen support for begin_access [ no_nested_conflict ] .
Faster comparisons with integer literals
Eager specilization shouldn ' t optimize functions with ownership ( yet )
LangRef : add some basic documentation for generics .
SIL : fix the hash function for Projection
[ Sema ] [ test ] Add SR - 13899 test cases
Resolve an issue with indirect enum ' s match for opaque values
[ ConstraintSystem ] Move missing contextual protocol detection to ` matchExistentialTypes `
Delay generation of ObjC protocol witnesses until the end of the outermost Clang import .
Merge pull request from slavapestov / generic - capture - promotion
[ DebuggerSupport ] Unbreak closures in the expression parser .
AST : Fix protocol composition canonicalization for primitive AnyObject
[ overlay ] Implement UIFocusEnvironment . contains using a shim
docs : add documentation for libdispatch on windows
[ AST ] Block recursion through ProtocolDecl : : getInheritedProtocols ( ) .
fix " newline @ end of file " warning .
Merge pull request from DougGregor / resilient - table - init - empty - proto
Be less conservative with enum initialization sequences
Third batch of changelog edits ( )
ArraySlice indexes no longer zero - based
Merge pull request from bitjammer / typeref - uniquing
Reorganization within RemoteAST . Basic error propagation .
Revert " stdlib : fix a warning "
Merge remote - tracking branch ' origin / master ' into master - next
Merge remote - tracking branch ' origin / master ' into master - next
change Type : : findIf and Type : : transform templates to be standard
Merge remote - tracking branch ' origin / master ' into master - next
[ ParseableInterface ] Don ' t print @ _hasInitialValue for resilient variables
Import C structs and C + + classes as Swift structs .
Merge remote - tracking branch ' origin / master ' into master - next
Add GenStruct . h , which I meant to commit with r1806 .
Update WindowsBuild . md
Merge remote - tracking branch ' origin / master ' into master - rebranch
Float to String benchmark : Use better workloads .
Only reflect NSString categories ( outside of Foundation ) on String .
stdlib / Dictionary : address internal API review : preallocate storage when
Merge remote - tracking branch ' origin / master ' into master - next
[ AST ] Do type / attr setup in ClassDecl : : addImplicitDestructor .
SE - 0138 : UnsafeRawBufferPointer revision .
Merge pull request from ravikandhadai / constexpr - debug - stdlib
Merge remote - tracking branch ' origin / master ' into master - next
Fixing tests for Linux
[ NFC ] Silence some Wnewline - eof violations
Compile swift . swift sources all at once , but not from catenated file
Rename swift : : Demangle : : archetypeName ( ) to swift : : Demangle : : genericParameterName ( ) .
Lazy construct the set of implicitly used generic params
Merge remote - tracking branch ' origin / master ' into master - next
[ ConstraintLocator ] Minor indentation fixes
Merge pull request from bob - wilson / llvm - r335407
Merge remote - tracking branch ' origin / master ' into master - next
Sema : Try harder to preserve ParenType sugar when when performing upcasts
Bridge NSArray < - > AnyObject [ ] .
some random cruft for a randomly crufty document .
Merge pull request from gottesmm / pr - 7883f67fb17b404bcbc2ef7b9028f09bbbaabe36
Merge pull request from compnerd / trailing - msvc
Merge pull request from jckarter / quartzcore - test - fix
Merge pull request from apple / revert - 11397 - simplify_performMemberLookup
push global reference parsing forward a bit , diagnosing a type error .
stdlib / String : fix a memory safety issue with using foreign indexes
Preliminary evolution tests
Make TypeChecker : : typeCheckExpressionConstraints ( ) more of a " real "
stdlib : use the destructor CC adjustment thunk
Don ' t pass - enable - guaranteed - self in tests
add missing newlines at end of files .
ClangImporter : Import blocks as [ objc_block ] types
Merge pull request from slavapestov / versioned - fixes
Update the call graph with newly linked - in functions .
[ Constraint system ] Minor fixes / test updates for function builders .
fixes
swift - api - digester : teach the tool to output constructors ' parameter type changes . ( )
[ test ] One more @ objc - > dynamic change .
[ NFC ] Pick a few nits for Jordan
[ SIL ] Witness methods store the conformance from which they come .
Make capture analysis handle assignments where the LHS is a tuple . < rdar : / / problem / 12425555 >
[ ClangImporter ] Split out the accessor parts of importMethodType .
Sema : Fix memory corruption in another circularity check
Correctly apply typechecking solutions with subscripts on type instances
Sema : Ensure that validateDecl ( ) is not called with imported or deserialized declarations
[ driver ] Use if / else instead of ternary , to please VS2017 .
[ IRGen ] Adjust for LLVM r373054
Remove unused NSRange slice variable ( )
Configure the swiftImageInspectionShared to generate arch & sdk specific libraries
Merge pull request from ffried / fix_comparison_typo
Merge pull request from practicalswift / swiftc - 28692 - unreachable - executed - at - swift - lib - ast - type - cpp - 1349
AST : Split off StorageImplInfo from AbstractStorageDecl : : AccessorRecord
Simplify : TupleExprs can ' t have null elements .
CollectionsMoveIndices : wording improvements , added an analysis section
[ frontend ] Add proper support for determining the InputKind .
CMake : set SDK , deployment target , and architecture for library variants .
[ cast - opt ] Refactor some code onto SILDynamicCast and delete dead code .
Eliminate extraneous insertion into ResolvedOverloads ; it ' s always
Merge remote - tracking branch ' origin / master ' into master - next
Merge remote - tracking branch ' origin / master ' into master - rebranch
Merge pull request from jckarter / metadata - reader - anonymous - context - size
[ ConstraintSystem ] Favor resolving closures over any disjunction binding type variables
Merge pull request from practicalswift / typo - fixes - 20160216b
Merge remote - tracking branch ' origin / master ' into master - next
[ sourcekitd ] Change compile notifications to pass a single args string
[ AddSwift . cmake ] Remove leftover use_internal_sdk parameter values
Merge pull request from ahoppen / SR - 2575 - 1 - ordering - DeclNames
[ CopyForwarding ] Make my comments clearer
Merge remote - tracking branch ' origin / master ' into master - next
PrintAsObjC : use getObjCPropertyName ( ) consistently for property name
Convert the diagnostic methods into passes .
[ AST ] Adjust for LLVM r373437
[ IDE ] Introduce ide : : printDeclUSR ( ) which uses the mangler to provide a USR for a swift symbol .
PrintAsObjC : Update for primitive AnyObject
AST : Optimize construction of the AnyObject dispatch table
Merge pull request from rintaro / sema - diag - valueofmeta
Merge remote - tracking branch ' origin / master ' into master - next
Merge remote - tracking branch ' origin / master ' into master - next
buildbot : Typo in install - test - script . sh
Merge remote - tracking branch ' origin / master ' into master - next
Merge pull request from xwu / nunc - est - dividendum
[ stdlib ] [ oslog ] Create an SPI _globalStringTablePointer that exposes
[ Android ] Enable building static stdlib on macOS
String : add some unavailable subscript overloads for String with integer ( or integer range ) indices .
At Chris ' s request , extract out some exposition into a document
Use getParams ( ) rather than getInput ( ) in rewriteType ( ) .
Extract the core implementation of ReflectionContext and
swift - module - digester : diagnose reordering of members in a fixed layout type under ABI mode .
Merge pull request from DougGregor / gsb - compare - dependent - types - cleanup
Copy legacy layouts to stdlib build directory
Merge remote - tracking branch ' origin / master ' into master - next
[ GSB ] Use a larger set for checking for visited requirements in ' isRecursive ' .
test / Frontend / dependencies . swift fails when the compiler is built
Merge pull request from xedin / SR_3506
AST : Fix a typo
Merge remote - tracking branch ' origin / master ' into master - next
Add test for the SwiftSyntax tree produced when parsing ' = super ' in a module interface
Merge pull request from rintaro / ide - completion - rdar58550697
Merge pull request from practicalswift / gardening - 20170121
Split the debugging feature to set the pass name inside the PassManager resetAndRemoveTransformations into a new function .
[ ConstraintSystem ] Defensive guard not to join unresolved type
Simplify and optimize the structural type metadata caches to use ConcurrentMap directly .
Delay import - as - member resolution even without an explicit swift_name .
Merge pull request from gottesmm / pr - dddd2694ec260afaf09db5b3ad66c9a296e6c666
Miscellaneous test migrations
Fix spaces in AST printer results for { get set }
[ NFC ] Don ' t crash when we see an empty filename for a module . ( )
Merge pull request from davezarzycki / exhaustive_coerceToType
Merge pull request from bob - wilson / update - llvm - r347806
Add radar number for disabled test .
[ serialization ] Introduce ModuleID , with codes for special modules .
Merge remote - tracking branch ' origin / master ' into master - llvm - swift5 - transition
Merge remote - tracking branch ' origin / master ' into master - next
Re - apply " Reduce memory footprint of the Swift compiler "
[ CSDiagnostics ] Find argument lists for key path subscripts ( )
Fixed issue in design document where incorrect word mentioned
stdlib : Add ' invoke : ' label to withVaList
Module groups : partially address @ gribozavr and @ natecook1000 ' s comments .
Merge pull request from atrick / silcombine - fix
Sema : Fix crash with metatype construction - vs - dynamic Self
Sema : Solve Bind constraints involving to two variables of different lvalue - ness
[ SyntaxTreeCreator ] Add documentation comments for the SyntaxTreeCreator class
Merge pull request from rudkx / ordering
[ stdlib ] Unify _NativeDictionary . subscript . _modify ’ s two yields to work around SR - 10604
[ gardening ] Silence a warning in release builds . ( )
[ sil - mode . el ] Add syntax highlighting for load_unowned / store_unowned .
[ allocbox - to - stack ] Improve comments . NFC .
Merge pull request from practicalswift / typo - fixes - 20160225
Merge pull request from eeckstein / osize - benchmarks
[ serialization ] Add missing headers to . cpp file .
[ CodeCompletion ] Calculate type relation for EnumElementDecl
[ Frontend ] Factor out getSourceFileParsingOptions
build : explicitly state arch for ICU
Add PR test presets for sourcekit - lsp and indexstore - db
[ RemoteAST ] Fix metadata reader to properly read tuple type metadata
[ AutoDiff upstream ] Update LoadableByAddress . ( )
improve type safety and specificity a bit , the body of a function is always a FuncExpr .
Mark tuple splat and + + / - - as errors instead of warnings . This
[ CodeCompletion ] Rename GenericParam kind to GenericRequiremnt
[ Runtime ] Lookup @ objc protocols in mangled name - > type metadata function .
[ migrator ] Fix offset in no_duplicate_aarch64_use_tbi test
[ SIL ] Add SILInstruction : : mayHaveSideEffects helper .
Merge pull request from lancep / CVarArgBoolConformance
Don ' t verify the AST in NDEBUG ( Release ) builds .
Merge pull request from rudkx / prefer - binding - disjunctions
Make matchFunctionTypes take a callback to match parameter and result types .
Merge remote - tracking branch ' origin / master ' into master - next
Merge remote - tracking branch ' origin / master ' into master - next
Merge pull request from DougGregor / witness - serialization
[ ast ] Define BuiltinTypeKind a derived enum from TypeKind that allows for exhaustive switching over builtin types .
Merge pull request from atrick / sil - verify - none
Rearrange test code for exhaustiveness
Fix the Makefile build
Don ' t define SWIFT_USE_SWIFTCALL if there is no such attribute
FunctionSignatureOpts : also specialize externally available functions .
Merge pull request from rintaro / ide - completion - rdar66002497
Serialization : Deserialize apply and partial_apply using substFnType ' s interface type .
Merge pull request from CodaFi / import - of - call
AliasAnalysis : better handling of init_enum_data_addr and init_existential_addr .
[ NFC ] Reorganize the AllocBoxToStackPass .
Merge pull request from nkcsgexi / closure
Merge remote - tracking branch ' origin / master ' into master - next
ChangeLog addition : SR - 7139 : Support enforcement of exclusive memory
Add some more execution tests for array downcasts of non - verbatim bridged types .
Merge pull request from shajrawi / licm_refelem
Merge remote - tracking branch ' apple / master ' into android - build_toolchain
[ Runtime ] Add test for casting an Error type to NSObject
[ OSLogOptimization ] Improve the ` replaceAndFixLifetimes ` function
[ Update checkout ] Add support for upstream - with - swift
[ code - completion ] Fix crash with missing func body in type - checking
[ ConstraintSystem ] Add a special locator for type of the key path expression
Revert " Dictionary test : fix for platforms that don ' t have return - autoreleased "
Resolve crasher 28704 ( )
[ IDE ] Correct uninitialized boolean
[ SR - 5856 ] Test case for protocol extensions stored properties ( )
[ swiftc ] Add test case for crash triggered in swift : : TypeChecker : : getTypeOfRValue ( swift : : ValueDecl * , bool )
IRGen : Implement ' x is T ' for archetype T .
Merge remote - tracking branch ' origin / master ' into master - next
[ ParseableInterface ] Add diagnostics test for errors in inlinable function bodies
[ test ] More type annotations for faster builds .
Merge pull request from davidungar / type - fingerprints - on - by - default - 5
[ Driver ] fix support for - embed - bitcode - marker .
[ String ] String . Index . init ( _ : within : ) bounds checks
ParserStatus : always initialize all members
Merge pull request from xedin / closure - arg - diag
Merge pull request from valeriyvan / FixExampleSnippetsSequenceAlgorithms . swift
test : decode the output of ` lldb - P ` as UTF - 8
[ Sema ] Use getMetatypeInstanceType instead of casting to MetatypeType
[ Constraint solver ] Remove invalid assert from constraint propagation .
Merge pull request from slavapestov / dynamic - stored - property - with - observers - fix
Properly specify dependencies for libAST and libParse
[ cmake ] Add the option SWIFT_BUILD_RUNTIME_WITH_HOST_LIBLTO .
Merge pull request from compnerd / storage
[ Standard library ] Drop _typeByMangledName ( ) .
[ NFC ] Add an EmitSortedSIL member to SILOptions for consistency . ( )
[ Stdlib ] Eliminate the _UnicodeEncoding_ protocol .
Wire up sema support for field access of single element data declarations . This allows Foo . X to work with structs .
stdlib : finish implementation of atomics for initializing an ARC
Build ASTs correctly for unqualified references to metatype members of a generic type .
Sema : Lazily validate protocol members
Merge remote - tracking branch ' origin / master ' into master - llvm - swift5 - transition
Merge pull request from ikesyo / if - os - macos
Merge remote - tracking branch ' origin / master ' into master - next
Override checking : believe full - name matches over types . ( )
Reinstate " Improve funciton signature @ owned return result to " not owned " conversion "
Merge pull request from CodaFi / mutatis - mutandis
Fix typo : correspondance → correspondence
Attach a foreign error convention to @ objc methods .
Fixup some tests I missed since sil - opt was not running with + 0 self enabled in all cases .
Remove unused header file .
dprintf ' s format argument is read - only
Merge remote - tracking branch ' origin / master ' into master - next
Merge pull request from eeckstein / option - set
Merge pull request from adrian - prantl / 31926379
[ devirtualization ] Begin refactor devirtualization code into first simple cases and then the general case .
[ runtime ] Add platform independent version of dladdr ( ) / Dl_info
[ docs ] LibraryEvolution : Move " Versioning Internal Decls " to the end .
Always allow the extension where a conformance is declared to be checked
[ Remarks ] Add a specialized RemarkStreamer for SIL remarks
libSyntax : parse codeblock syntax node . ( )
Merge pull request from eeckstein / disable - lazy
Merge pull request from rintaro / refactoring - rdar42098130
Merge remote - tracking branch ' origin / master ' into master - next
Merge pull request from eeckstein / string - enums
[ test ] Add Range overlaps benchmark
Merge pull request from nkcsgexi / preset
Merge remote - tracking branch ' origin / master ' into master - next
XCTest Overlay : Treat ObjC exceptions as unexpected failures
Reformat doxygen comment .
Merge pull request from practicalswift / fix - pep8 - regression
Merge pull request from apple / remove - todo - wip
Remove since identical with 00805 - swift - constraints - constraintsystem - opengeneric . swift .
[ pred - memopt ] Rather than extracting values at the load site , extract at the store site .
Merge remote - tracking branch ' origin / master ' into master - next
Peephole forces of certain optional - to - optional conversions .
stdlib / String : add tests for trapping on out - of - range String . UTF16View indexes
Switch NecessaryBindings to use PolymorphicConvention , making it much lazier .
[ sil ] Add support for adding _semantics attributes to nominal types .
[ Gardening ] Clean Up OS - Test Patterns Across The Codebase
[ Constraint solver ] Don ' t remove / reintroduce disjunction when favoring .
Merge remote - tracking branch ' origin / master ' into master - rebranch
Small control - flow simplifications , NFC .
[ Diagnostics ] Adding optional note specific to keypath application base diagnostics
SIL : Fix visibility of final method symbols in resilient classes
DI : Factor out LifetimeChecker : : emitSelfConsumedDiagnostic ( )
Merge pull request from DougGregor / no - assoc - type - override - witnesses
Allow allocating a type of any ArrayRef from ASTContext .
Remove ImplicitlyUnwrappedOptionalType .
Linux cmake script env doesn ' t have pushd / popd .
Merge pull request from milseman / uniguard
Merge remote - tracking branch ' origin / main ' into next
Merge pull request from compnerd / fix - docs
Merge pull request from bitjammer / rdar - 32213595 - migrator - fixitapply - invalid - sourceloc
Merge pull request from xedin / rdar - 40002266 - reapply
Merge pull request from rudkx / minor - cleanup
IRGen : When available , use ClassMetadataStrategy : : Update even if we don ' t have legacy type info
Update YAML format
[ gardening ] Use consistent end - of - namespace comments
[ Parse ] Fix issue with incremental re - parsing for SwiftSyntax emitting bogus parsing error
Clang importer : compare initializers by their parameter types , only .
Parse : Record which delayed type and extension bodies possibly contain operators and classes
[ Async CC ] Added task and executor args .
[ ConstraintSystem ] Make ` ForceOptional ` fix a part of ` repairFailures `
Add NewReferenceExpr , for allocating class objects .
AST : Fix name lookup from within lazy property initializers
Merge pull request from DougGregor / stdlib - se - 0142
stdlib : AnyHashable tests : finish tests for CF types
Update README with new Xcode information ( 10 . 2 beta )
Remove trampolines for using concrete hashable types with Set / Dictionary of AnyHashable
[ windows ] Compare SWIFT_HOST_VARIANT against lowercase value .
Merge remote - tracking branch ' origin / master ' into master - next
random tidying
Merge remote - tracking branch ' origin / master ' into master - next
Merge remote - tracking branch ' origin / master ' into master - next
Merge pull request from slavapestov / more - class - layout
Selector splitting : the first parameter name is always relevant .
Merge remote - tracking branch ' origin / master ' into master - next
SILOptimizer : Fix Mandatory Inlining problem with opened existentials
Replaced assert ( false with llvm_unreachable
Simplify configure pass for Foundation to use a defined variable for XCTest ' s build directory
Factor constraint diagnostic search functionality out of the salvage method .
Merge pull request from aschwaighofer / silgen_fix_withoutActuallyEscaping_c_closure
[ benchmark ] ReportFormatter : better inline headers
Merge pull request from jtbandes / master
Integer overflow diagnostics : specialize diagnostic message for underflow
Test reconstruction of non - generic nested types with contextual where clauses
Merge remote - tracking branch ' origin / master ' into master - next
Add some helper methods . NFC .
[ benchmark ] SetTests : Move legacy benchmarks to end of list .
Stop pretending that the layout of swift strings are opaque magic
Move ` build - script ` apply default arguments into separate module ( )
[ ClangImporter ] " Handle " the new ObjCTypeParamType . ( )
Revert " Switch MetadataCache to use a global slab allocator . "
[ CSDiagnostics ] Small cleanup in tryProtocolConformanceFixIt ( )
[ cse ] Add support for global_addr .
[ ConstraintSystem ] Reinstate favoring more specialized of two generic overloads .
Add a flag to allow Swift objects ( such as the singleton empty collections ) to ignore refcounting
SIL serialization : add PrettyStackTraces ( )
Fix linux test
Merge remote - tracking branch ' origin / master ' into master - next
Add support for emitting parseable - outpuit " began " message from swift - frontend
Move Bool out of Misc . swift
Sema : Remove CSApply ' s getNaturalArgumentCount ( ) in favor of getNumCurryLevels ( )
Format doxygen comment correctly .
clang - format the file .
Make preposition list actually match what I ' m proposing
[ SE - 0160 ] @ IBInspectable implies @ objc
Add an unsupported error for curried local functions for the moment .
[ sil - devirtualizer ] Re - factoring making it possible to check if an apply can be devirtualized . NFC .
[ Typechecker ] Remove period from ' end anonymous namespace '
Merge pull request from xedin / switch - to - use - generic - signature - of - context
Merge remote - tracking branch ' origin / master ' into master - next
Merge remote - tracking branch ' origin / master ' into master - next
Implement the first half of SE - 0109 : Remove the Boolean protocol
Implement support for type - attributes on the result of a function type or decl .
Merge pull request from practicalswift / swiftc - 28325 - swift - typebase - getcanonicaltype
[ Clang importer ] Lazily synthesize union field getter / setter bodies
Fixing validation tests
[ llvm - arc - opts ] Refactor the ARC call constant caches and ARC call creation code into a builder class .
Give CanType a getAnyOptionalObjectType accessor .
SILGen : Emit generic dynamic casts .
SILGen : Implement getting properties .
Merge remote - tracking branch ' origin / master ' into master - next
[ Concurrency ] Use SchedulerPrivate for the " next waiting task " link .
Merge pull request from slavapestov / decl - checker - cleanup - part - 3
[ core ] hasPrefix / hasSuffix consider the empty string a prefix / suffix of all strings .
More tests
Merge remote - tracking branch ' origin / master ' into master - next
Abbreviate the constraint argument names to match the comments . NFC .
Revert " Reenable swap_refcnt . swift test . "
Merge remote - tracking branch ' origin / master ' into master - next
Manually merge origin / main into origin / next .
[ SILGen ] Emit all TopLevelCodeDecls in a single scope .
[ Runtime ] Android does not support constexpr
fix a warning about a zero length array , by ripping out all of the old declattributes code . : - )
build : remove ` TARGET_SDKS ` from the runtime
Add some tests for @ testable and subclassing .
Add comment and renamed lookUpTopLevelNames …
[ coverage ] Add a preset for generating coverage data and have coverage - generate - data script use that .
[ github ] Document what the macOS / Linux pull requests do in the PR template for easy access .
[ Foundation ] Rename swift_errorBridgingInfo so it doesn ' t conflict with pre - stable ABI Swift
Merge remote - tracking branch ' origin / master ' into master - next
Change from SWIFT_BUILD_ROOT to SWIFT_BUILD_DIR in examples . SWIFT_BUILD_ROOT means something else in the build script .
Merge typevars of operators with shared decls after finding a solution to speed up further searching
Merge remote - tracking branch ' origin / master ' into master - next
[ AST ] Make dumping for SubstitutionMap lisp - formatted and indent - aware .
Merge pull request from swiftix / partial - specialization - fixes
Merge remote - tracking branch ' origin / master ' into master - next
Merge pull request from slavapestov / old - sema - flag
AST type printer : add testing infrastructure and a basic test
[ load - store - opts ] Refactor out call to recursivelyDeleteTriviallyDeadInstructions - > LSBBForwarder : : deleteInstruction .
[ QoI ] Switch to diagnoseMemberFailures for unresolved members
[ GSB ] Eliminate PotentialArchetype from enumerateRequirements ( ) .
Merge pull request from nkcsgexi / custom - buffer
Remove the ' - disable - all - runtime - checks ' option that predates current SIL - level
Merge remote - tracking branch ' origin / master ' into master - next
[ nfc ] Remove some unused SILBuilder methods .
Merge pull request from jckarter / optional - bridging
Improve testcase
refactoring : Add operator for comparing OperandValueArrayRefs .
Merge pull request from zoecarver / cxx / benchmark / create - objects
Make Float / Double be typealiases to Float32 / Float64
Merge remote - tracking branch ' origin / master ' into master - next
Fix some validation - test regressions , mostly by tweaking tests
[ Foundation ] Extract NSTextCheckingResult extensions into a seperate file
swift - module - digester : refactor all node creation functions into SwiftDeclCollector . NFC
Sema : Reference ' Self ' archetypes back to their AssociatedTypeDecl .
Added PlaygroundTransform tests on implicit return .
Implement let declarations .
Add fix - its for missing and spurious separators .
Merge remote - tracking branch ' origin / master ' into master - next
Silence a warning about unused an unused debug function .
Merge pull request from DougGregor / name - lookup - debugging - cruft
Replace materializeForSet with the modify coroutine .
Merge remote - tracking branch ' origin / master ' into master - next
Fix typo in comment .
Fix for extensions in ProtocolConformanceAnalysis
Framework for builtin lookup .
Merge pull request from practicalswift / swiftc - 28353 - swift - removeoverriddendecls
Cache the isTrivial property .
Change the playground transform instrumenter such that it provides the logging calls direct access to the source locations
SIL : Lower DynamicSelf . metatype down to the metatype of the self type .
Previous fix was actually for rdar : / / 19478919 , correct the testcase .
[ incrParse ] Reparse a node if the next leaf node has been modified
. . and also fix a few that the script didn ' t seem to find
runtime : namespace SmallVectorImpl for inline namespace
IRGen : Split up class metadata emission into fixed and resilient cases
Merge pull request from atrick / opt - escape - endaccess
Merge remote - tracking branch ' origin / master ' into master - next
implement support for Ty . member to represent the same thing as Ty : : member .
Merge pull request from slavapestov / fix - field - offset - global - update
Merge remote - tracking branch ' origin / master ' into master - next
Disable tsan test on tvOS pending investigation .
[ AST ] Have the ASTWalker visit the indices pattern and element TypeRepr of a subscript decl .
[ swiftc ( 43 vs . 5395 ) ] Add crasher in swift : : LValueType : : get ( . . . )
Disable test / Migrator / rdar31892850 . swift
Merge remote - tracking branch ' origin / master ' into master - next
SILCombine : optimize creating enums with tuple payloads .
lit . cfg : fix a bug in simulator target
[ gardening ] Eliminate unused variable warnings from non - asserts build .
Merge pull request from practicalswift / cls - fix
Merge pull request from rudkx / propagate - constraints
Update some tests so that they pass ownership verification .
Move a few more type checker perf tests over to compiling with designated types .
Merge remote - tracking branch ' origin / master ' into master - next
pull in StringRef and Twine .
Merge pull request from trentxintong / RCCache
[ stdlib ] Give Array bridging a home of its own
Add several AccessedStorageAnalysis unit tests , and fix iterator invalidation .
Merge pull request from lanza / lgone
runtime : Fix a compiler warning in release the build
Merge remote - tracking branch ' origin / master ' into master - next
CollectionsMoveIndices : eliminate IndexType , it does not have any requirements
[ docs ] Add " iff " to Lexicon . rst .
Merge pull request from harlanhaskins / profdata - merge - fix
Merge pull request from rjmccall / tentative - custom - attr - parsing
Merge remote - tracking branch ' origin / master ' into master - next
Merge pull request from benlangmuir / context - match - rebalance
Merge pull request from aschwaighofer / silgen_soften_assertion_in_project_open_opaque_existential
Merge pull request from lorentey / romannumbers
Merge pull request from DougGregor / changelog - se - 0157
Support generic return types , as long as they don ' t differ
[ tests ] Fix diagnostic ( TODO )
Add - - enable - throw - without - try option to suppress warning about missing ' try ' .
Don ' t emit debug info for generic types with no declcontext .
Fix stdlib deployment targets for powerpc64 and powerpc64le
[ gardening ] Fix unjustified spacing
Release note the CF import change .
Unwrap params to satisfy older SDKs where observeValueForKeyPath takes non - optional
Merge remote - tracking branch ' origin / master ' into master - next
Merge pull request from shajrawi / try_apply_bugfix
add a new LiteralExpr : : shallowClone method , to be used by upcoming work . NFC .
[ Serialization ] Extract error types to a separate header .
SILGen : Don ' t expect an override for declarations from class extensions .
Merge pull request from eeckstein / fix - identifier - mangling
SILOptimizer : fix a stupid bug in StackNesting which can cause a miscompile in functions with unreachable blocks .
AST : Remove some unnecessary calls to setGenericSignature ( )
Rename Devirtualizer . cpp to SpeculativeDevirtualizer . cpp .
[ + 0 - all - args ] Teach SwitchEnumBuilder how to handle cases with a jump dest continuation .
Re - add the optional overload for atomic ptr swaps
Merge pull request from austinzheng / az - gardening
testcase fixed by recent changes .
Add a test for - disable - typo - correction
IRGen : Change initial size of a SmallVector , NFC
Made Decl ' s ClangNode storage support 32 - bit platforms .
Make the python program sort unicode strings to match the ObjC and Swift implementation .
Merge pull request from drodriguez / executable - stored - properties - client
[ Docs ] Convert TypeChecker . rst to Markdown
[ InterfaceGen ] Print property initializers in resilient , fixed - layout types ( )
revert _CocoaDictionaryBuffer indexing design
Cache the dependence of types in RecursiveTypeProperties . NFC .
Merge remote - tracking branch ' origin / master ' into master - next
[ benchmark ] DataBenchmarks Legacy Factor
Don ' t bind generic type parameters to non - materializable types .
Fix library_IMPLIB - NOTFOUND errors building Swift on Windows
Fix typo : dependences → dependencies
[ Foundation ] Extracting the ReferenceConvertible conformance for CharacterSet ( )
Update the pattern - matching proposal after thinking it over for a
Put the requires back in its place
Staging for __consuming
Add UncheckedOptional < T > even to special typedef types .
[ stdlib ] Update revisions with feedback
[ SIL ] Add test case for crash triggered in swift : : Parser : : parseSILWitnessTable ( )
Don ' t quote DOTEST_EXTRA as they ' re part of a CMake list
[ stdlib ] swift_isClassOrObjCExistentialImpl fix
Merge pull request from egorzhdan / winsdk - console
[ typechecker ] remove null check as its redundant
Update LangRef with commit message of r17056
Add a new " mark_function_escape " instruction that will be used to mark the
A conformance is non - inheritable if it has a non - abstract initializer witness .
[ Sema ] Match Swift 4 . 0 / 4 . 1 overloading behaviour for properties in extensions of generic types .
[ android ] Add support for natively building on Android
Merge pull request from davezarzycki / nfc_hasParenSugar
Merge pull request from slavapestov / small - existential - type - cleanup
[ cmake ] Do not use LLVM ' s configure_lit_site_cfg to configure out lit config .
Don ' t walk into an as - of - yet - unsynthesized function body .
AliasAnalysis : code reformatting and fixing comments .
Merge pull request from rjmccall / mutating - reqt - overrides
Merge pull request from compnerd / 29582 - 1
[ QoI ] Improve fixit for missing parameter clause in initializer decl ( )
Merge pull request from smeenai / master
[ Omit needless words ] Don ' t split subscript accessors .
[ build - script ] Forward cross - compile host flags to lldb ( )
Simplify the check and fix a bug .
stdlib : remove Keyboard
[ SILGen ] Handle Reabstraction and Substitution in assign_by_wrapper
New indexing model : replace a wrong range check with a FIXME
Add a - print - clang - stats frontend flag .
[ Type witness inference ] Use “ local ” dependent member types when possible .
Merge remote - tracking branch ' origin / master ' into master - next
Guard against passing endIndex to foreignScalarAlign when back - deploying to 5 . 0 stdlibs
split CFGLowering out to its own sub - library of the CFG library .
Merge remote - tracking branch ' origin / master ' into master - next
Rewrite MetadataCache to be a more natural extension of ConcurrentMap .
Add a CHANGELOG entry for [ SE - 0054 ] .
[ Constraint application ] Handle materialization for implicit byrefs .
Merge pull request from compnerd / frontend - tests
Merge remote - tracking branch ' origin / master ' into master - next
swift - reflection - dump : use elaborated type name
Incorporate Doug ' s review suggestions
Basic vim highlighting for Swift .
[ NFC ] Used convenience to retrieve context arg .
Add a TypeInfo method to UnimplementedTypeInfo that snuck in .
Merge pull request from rudkx / clear - opaque - expr - on - open - existential
[ stdlibunittest ] Still more de - boilerplating WIP
Merge pull request from apple / stdlib - coding - style
fix ICU include paths . Please check this commit and verify that the previous solution simply coudn ' t have worked * at all * . The usage of slightly differently named variables ( e . g . * LIB vs * LIBS ) and lower - case module in the include for loop while a TOUPPER variable should have been used , caused the INCLUDE variables to never be filled . Apart from that , a find_package ( ) was missing in the stubs CMakeFiles . txt .
[ build - script ] Ensure we have all of LLDB ' s test dependencies ( 2 / 2 )
Merge remote - tracking branch ' origin / main ' into next
[ Mangler ] Fix a bug in the bit width calculation of numbers representing strings .
Revert " [ CMake ] Fix up and fill out swift exports . "
[ Testing ] Unbreak Linux builds that do not have the scudo runtime
stdlib : Dictionary : contain the unsigned division optimization in a getter method , expose a clean API to the outside
Merge pull request from rintaro / gyb - doctest
[ Serialization ] Remove " delayed actions " support
Merge remote - tracking branch ' origin / master ' into master - rebranch
Merge remote - tracking branch ' origin / master ' into master - next
Clang importer : allow subsequent macro definitions to replace earlier ones .
[ concurrency ] IRGen : : do ptrauth for hop_to_executor
Merge pull request from gottesmm / pr - 87b3cae407e80a1fb5e3ad4de714847cf43d0439
Merge remote - tracking branch ' origin / master ' into master - next
Remove _interface
[ SE - 0258 ] Properly contextualize closures in property wrapper initialization .
Merge pull request from xedin / restore - cs - debug - output
[ codecomplete ] Check for ErrorType before asking for CD - > getResultType ( )
Merge pull request from slavapestov / inline - always - is - not - inlinable
Sema : Don ' t clone TypeReprs in ConstraintSystem : : shrink ( )
EscapeAnalysis : make a separate state for " escapes with return "
Merge pull request from xedin / rdar - 50666427
[ stdlib ] BridgeStorage optimizations
Merge pull request from aschwaighofer / changelog_entries
Merge remote - tracking branch ' origin / master ' into master - next
remove compiler support for let / else . This simplifies PBD back to being
Merge remote - tracking branch ' origin / master ' into master - next
fix setGenericSignature
Fix < rdar : / / problem / 16906000 > Implicitly unwrapped optional let is not considered initialized , but var is
Merge remote - tracking branch ' origin / main ' into next
Setters default to mutating .
Merge remote - tracking branch ' origin / master ' into master - next
[ Exclusivity ] Add deserialization callbacks from the optimizer .
Merge remote - tracking branch ' origin / master ' into master - next
[ OSLogOptimization ] Improve the OSLogOptimization pass so that it can
Merge remote - tracking branch ' origin / master ' into master - next
Merge pull request from gottesmm / pr - b925e2e08f6a48880a581d4103bfeae1b80a2a5c
Generalize UnsafeBufferPointer - as - a - Collection testing .
Merge pull request from graydon / batch - mode - pch - attach - diagnostic - multiplexing
Merge pull request from fredriss / toolchain - spring - cleaning
fix a merge issue which broke r31577
Merge pull request from gottesmm / pr - c948d27bcce9be4feb87ece1fc46b74931415542
Merge remote - tracking branch ' origin / master ' into master - next
Pull the rest of top - level type resolution into its dedicated function . NFC
Merge pull request from gottesmm / checked_cast_br_1
Merge pull request from vermont42 / literal - values
Merge pull request from jutaz / bugfix / typos - in - docs
remove an unnecessary # include in the middle of a file .
Merge pull request from DougGregor / remove - witness - table - accessor
stdlib : implement _stdlib_create_pthread_block in terms of Windows threaing
SILGen : Tidy up ManagedValue .
[ PrintAsObjC ] Use ' unsafe_unretained ' to print ' unowned ' , not ' assign '
fix indentation error
Merge pull request from apple / stdlib - string - comparable - move
Merge pull request from davidungar / fix - overly - conservative - fingerprint - bug
Fix unused variable warnings in release builds . ( )
[ Statement checker ] Always wire up case vars with bindSwitchCasePatternVars
Add SIL support for [ dynamic_replacement_for : ] functions
AST : Add ProtocolConformanceRef : : mapConformanceOutOfContext ( )
benchmark : Build drivers with N_AST symtab entries ( for debugging ) ( )
stdlib : fix coding style
Codesign test / Interpreter
Merge remote - tracking branch ' origin / master ' into master - next
Merge pull request from slavapestov / swift - ide - test - parser - lookup
[ ConstraintSystem ] Record ` unable to infer base ` only if hole originated from affected reference
Merge pull request from slavapestov / cleanup - for - lazy - accessors
Account for types that explode to sizes ! = 1 when computing the
[ IRGen ] Add main ( ) to __swift5_entry .
[ gardening ] Fix recently introduced typos
Merge remote - tracking branch ' origin / master ' into master - next
[ Sema ] Remove on - demand type checking for return type of ClosureExpr
Merge remote - tracking branch ' origin / master ' into master - next
Merge remote - tracking branch ' origin / master ' into master - next
[ ConstraintSystem ] When we have multiple conversions / fixes , make equality favored .
Add test for witness method devirtualization
Merge pull request from compnerd / swift - is - swiftc
SILGen : Add test to ensure we don ' t have a vtable entry for synthesized members of a final class
[ swiftc ( 50 vs . 5396 ) ] Add crasher in swift : : ASTVisitor
Revert " Hook instruction insertion & removal . "
Merge pull request from compnerd / rss - feeds
[ GSB / IRGen ] Allow redundant inheritance of the JSExport protocol .
[ docs ] Add OptimizationTips to the contents page .
[ Syntax ] Allow UnknownSyntax to have children
[ SIL Cloner ] Only substitute opened archetypes into conformances when needed .
Merge pull request from compnerd / prefix - suffix
[ Parse / ParsedRawSyntaxRecorder ] In recordToken ( ) adjust the range to account for backticks in an identifier
Merge remote - tracking branch ' origin / master ' into master - next
Expanded discussion of protocol types ( )
IRGen : Split up WitnessTableBuilder into { Fragile , Resilient } WitnessTableBuilder
add a dumper for effect Classification for debugging
Fix MemoryBehavior on SIL instructions that have a CastConsumptionKind .
[ benchmark ] Naming Convention
[ overflow ] Remove signed and unsinged overflow checks on addition of a stronger check was executed before the current one .
Merge pull request from rudkx / improve - too - complex - metric
[ Sema ] Move AccessScopeChecker classes to separate file in AST .
IRGen : Fix overly - specific enum_resilience test
[ GSB ] Eliminate redundant code in self - derived source checking .
Re - apply " Don ' t force - load Foundation under ImportForwardDeclarations . "
Reflection : Move some code from TypeRefBuilder . h to TypeRefBuilder . cpp , NFC
Demangler : Keep printing printable chars in QuotedString
We should SILCombine a checked_cast_br of metatype AnyObject . Protocol to Some . type to a direct branch
AST : Fix TypeBase : : adjustSuperclassMemberDeclType ( ) to strip generic signatures
[ aa ] Refactor all projection path logic into a class called ProjectionPath .
[ stdlib ] Fix Collection discussion typo
Sema : Replace trivial FunctionType : : getOld ( ) with decomposeInput ( )
Merge remote - tracking branch ' origin / master ' into master - rebranch
DeadStoreElimination : add debug dump functions
Merge pull request from DougGregor / resolve - type - null - check
SIL : Add a ClosureInst instruction type .
Update returnIfNotEqualGuard comment
Sema : Fix crash if explicit @ objc with a name is put on a deinitializer
[ Generic signature builder ] Substitute requirements instead of threading a PA around .
Merge pull request from lorentey / resilient - hashing
[ stdlib / private ] [ os log ] Simplify the OSLogPrototype stdlib - private
fix some test failures , which were introduced with the function - level optimization modes
Merge remote - tracking branch ' origin / master ' into master - next
Merge pull request from jtbandes / se - 0121
[ sil - devirtualizer ] Significantly improve devirtualization of inherited and specialized protocol conformances .
Merge pull request from apple / rxwei - patch - 1
[ gardening ] Use consistent header formatting .
[ test ] Internalize SmallString ; NFC
[ SymbolGraph ] Print non - underscored attributes in declarations
Merge pull request from jmittert / CloseHandle
Merge remote - tracking branch ' origin / master ' into master - next
Another Linux test fix for r32644
Fix IRGen for associated type witness accessor functions in generic contexts with
[ Reflection ] Substitute generics and resolve dependent members
[ ASTPrinter ] Fix issue where printing if - let variables shows their type as optional
Rename test .
[ stdlib ] Repair Substring - > String conversion
[ Property Wrappers ] Inject the opaque value placeholder for a property
Switch override checking over to interface types .
Add clang - cl to list of possible C / C + + compilers for Windows
stdlib / FixedPoint : fix strange indentation
Move sil - parser test projection_lowered_type_parse . sil to the correct location in tree .
Make the AST " phase " part of ASTContext , this is generally useful for assertions .
stdlib : add a hook for dynamic dispatch in CollectionType . find ( )
Added : Test case for the renamed to one of the overload methods case
Merge remote - tracking branch ' origin / master ' into master - next
add a comment .
Merge remote - tracking branch ' origin / master ' into master - next
KeywordArguments proposal : Describe vararg keywords , note QoI issue with ' foo . bar : '
Merge remote - tracking branch ' origin / master ' into master - next
[ test / Index / Store ] Remove the XFAILs for linux on some tests
Merge pull request from rudkx / remove - disjunction - number
[ unittests / Sema ] Add an ability to retrieve stdlib types by name
Merge remote - tracking branch ' origin / master ' into master - next
AST : Don ' t compute overload signature unless needed in removeShadowedDecls ( )
Fix build errors with clang - 421 .
[ gardening ] Fix recently introduced typo : " correctlly " → " correctly "
Merge remote - tracking branch ' origin / master ' into master - next
Merge remote - tracking branch ' origin / master ' into master - next
Merge pull request from swiftix / wip - opened - archetypes - v3
[ benchmark ] Extract yield & add resetMeasurements
fix formating in - ast - dump
Improved the performance of _dictionaryBridgeFromObjectiveCConditional by specifying the minimumCapacity up front . NFC
Implement grabbing an external value witness table . I
SIL Parser : parse builtin_zero , struct , struct_element_addr , and struct_extract .
Reduce boilerplate in the GitHub PR template
Make assert ( ) work correctly in debug and release builds respectively
Merge pull request from aschwaighofer / int_init_objectidentifier_label
Remove inadvertantly committed debug code from r27120 .
Merge remote - tracking branch ' origin / master ' into master - next
Add formerly - crashing test case from rdar : / / problem / 19840527 .
Define implicit struct constructors on demand and not lazily .
switch expression printing over to use a visitor instead of hand rolled dispatch .
Separate the computation of substitutions from the creation of SpecializeExpr .
Merge pull request from jckarter / replace - opaque - underlying - type - with - underlying - type
[ ClangImporter ] Import typedefs of block types as fully - bridged closures .
Merge remote - tracking branch ' origin / master ' into master - next
Merge pull request from tom - gall / gentoo - linux - fixes
[ Parse ] Parse trailing where clauses on associated types .
[ swiftc ] Add 💥 case ( 😢 → 55 , 😀 → 5098 ) triggered in swift : : GenericFunctionType : : get ( … )
Merge pull request from danra / patch - 30
Merge remote - tracking branch ' origin / master ' into master - next
Add a document that describes the design of the Swift Optimizer .
stdlib : specify to the optimizer that the capacity value of Dictionary / Set is never negative .
Fix < rdar : / / problem / 16411449 > ownership qualifiers don ' t work with non - mutating struct property
Use swift_unknownRelease instead of swift_release in swift_bridgeNonVerbatimFromObjectiveC .
Revert " CMake : build all platforms except watchOS using the public SDK "
[ ownership ] Add a few test cases that show that the reborrow verifier handles recursive reborrows correctly .
Merge remote - tracking branch ' origin / master ' into master - llvm - swift5 - transition
Python lint update - checkout
IRGen : Don ' t lazily emit witness tables .
Merge pull request from xedin / rdar - 31898542
WIP
[ SILOptimizer ] Add ‘ llvm . cttz . * ’ Intrinsic support to Constant Folding
Move convertLiteral ( ) into the solution application class .
sketch some more infrastructure for inout promotion , no functionality yet .
Merge remote - tracking branch ' origin / main ' into next
Merge pull request from practicalswift / remove - unused - function - convertExternalDefinitionIntoDeclaration
Fix typo in Remote Mirror build preset
[ android ] Add a test for Android conditional compilation ( AArch64 )
[ mandatory - inlining ] When using the linear lifetime checker to insert compensating releases , if we find a double use due to a loop , do not insert an apply at that call site .
[ SourceKit ] Initialize pointer as nullptr to fix a crash . rdar : / / 28959889
[ ModulePrint ] Remove relics of containing return types when printing optional init .
Merge pull request from gregomni / switch - cases
Fix diagnostic when we don ' t encounter a ' ( ' after parsing ' constructor ' .
Merge pull request from gottesmm / diverse_stack_improvements
IRGen : Enable usage of attributes from RuntimeFunctions . def
add a new Subsystems . h file to hold the entrypoints for various subsystems ,
Merge remote - tracking branch ' origin / master ' into master - next
Merge pull request from CodaFi / protocol - dictates - action
[ CodeCompletion ] Add an already - fixed crashing test case
Merge pull request from CodaFi / instructure
Sema : Remove even more vestigial TypeChecker usages
Update comments and assert messages in call graph verifier .
Merge pull request from compnerd / shell - is - not - parseable
[ Serialization ] Don ' t walk into function bodies for doc comments ( )
Merge pull request from practicalswift / american - english
swift - reflection - dump : Expected more
[ SILLLVMGen ] Fix a stack - use - after - scope
Merge pull request from aschwaighofer / irgen_fix_closure_assert
SILGen : Remove shouldReferenceForeignAccessor ( )
[ IRGen ] Foreign metadata is lazy metadata , always .
Remove debugging code
This test requires Objective - C interop
Merge pull request from DougGregor / gsb - cleanups
Mark invalid decls as erroneous and improve error location reporting
Merge pull request from rintaro / ide - completion - rdar57194358
[ benchmark ] PopFront Legacy Factor
Revert " [ stdlib ] Dictionary . updateValue ( _ : , forKey : ) : Don ’ t overwrite the existing key "
[ swiftc ( 25 vs . 5563 ) ] Add crasher in swift : : TypeChecker : : substMemberTypeWithBase
remove support for " func foo : : bar ( ) { } " . " Just use extensions " .
Sema : Simplify enum element validation
[ func - sig - opts ] When creating a new argument , pass in the decl from the old argument .
Further Vector - > Array migration . Also updated Array with several good suggestions from Dave A .
Merge pull request from compnerd / isysroot - is - isystem
[ Clang Importer ] Refactor out SwiftPrivate logic
Rewrite unused SIL pass descriptions as proper , descriptive pass names .
Always create initializer contexts for pattern binding entries in non - local scopes .
Merge pull request from gottesmm / eliminate_unnecessary_inout_constref_foreignSelf
Fix a typo ( thanks @ KentaKudo )
Disable Integers . swift . gyb test since it is failing on the bots .
EscapeAnalysis : improve the ConnectionGraph ABI .
Introduce a basic worklist into the constraint solver .
Make the SILDeclRef for a destroying destructor store a DestructorDecl .
[ stdlib ] Update documentation for case mappings
add a new alloc_tmp instruction , stub out ( not fully implemented ) alloc_var .
Call a standard library entry point from initializer stubs .
[ lit ] Add a more discoverable error message when failing due to attempting to perform an oss executable lit test run .
Update ARCOptimization . rst ( )
[ ConstraintSystem ] NFC : Extract logic related to forming locator for argument info retrieval
rename members of Instruction class to follow naming convention of ivars in the
Merge pull request from stephencelis / patch - 1
Merge remote - tracking branch ' origin / master ' into master - next
Merge remote - tracking branch ' origin / master ' into master - next
Merge pull request from ravikandhadai / LargeIntOverflowPR
Count AST ancestor scopes instead of children when last expanded
Reformat to reduce indentation .
[ AST ] Make dumping a recursive conformance not a horrible mistake .
Rename getInOutOrLValueObjectType to getWithoutSpecifierType
Merge pull request from DougGregor / minor - cleanups
Merge pull request from DougGregor / runtime - find - conforming - superclass
Heap : better hash function names and better hashing
[ cse ] Move subsystems header to be first since it defines the interface to the file .
[ AST ] NFC : Formalize Decl validation tracking via RAII
Loosen an assert .
Merge remote - tracking branch ' origin / master ' into master - next
[ frontend ] Add support for the - g option .
Merge remote - tracking branch ' origin / master ' into master - next
Merge pull request from xedin / SR - 2164
Move SILFunctionTypeInfo logic into SILType . cpp out of SILModule . cpp ,
Revert " Add regression test for rdar : / / 59496033 "
Merge pull request from rajbarik / raj - es - fix - default - enable
Move Default * and Minimal * colections to StdlibCollectionUnittest
Move PrintTypeRef and TypeRefSubstitution into the cpp - NFC
Clean up the command line interface for ' swift '
Remove now unnecessary const_cast
[ gardening ] Reduce indentation of SILGenLValue : : visitRec by inverting an if condition .
Added ZExt attribute for isUniquelyReferenced runtime calls .
TrapTestPrototype : relax CHECK line to allow this test to pass on iOS
[ NamedLazyMemberLoading ] Handle partial population from extensions better .
Conflict resolution and conversion to ConstraintFix class
Merge pull request from nvkiet / fix_typo
Merge pull request from aschwaighofer / sr4253_irgen_partial_apply_fix
[ arc ] Wire up ProgramTerminationFunctionInfo into ARCSequenceOpts .
Produce warnings when implicit optional promotions are introduced in some
Document the added lit substitution swift_src_root
lookupVisibleDecls ( ) : now we don ' t find references to generic types outside of
Merge pull request from amraboelela / android - build - toolchain2
Merge pull request from aschwaighofer / defer_body_not_dynamic
Merge pull request from shajrawi / explode_large_types_fix
Merge pull request from rintaro / lex - multiline - nlescape - last
lit : prefer ` discard ` over ` remove `
Properly pass the arguments specified for subscripting access down to
[ CodeCompletion ] Don ' t attach attr to decl if blank line exists
[ ReconstructType ] Remove a call to printf
Don ' t do is_trivial_constrictible on libstdc + + .
Remove - swift - version 3 lit substitutions
Merge remote - tracking branch ' origin / master ' into master - rebranch
[ Mangling ] Update SILGen and IRGen tests for mangling change .
Revert " Disable test / IDE / complete_from_clang_framework . It is failing when the stdlib is built without assertions . "
Switch the AST printer over to getLocalProtocols ( ) .
Merge remote - tracking branch ' origin / master ' into master - next
Merge pull request from rjmccall / rip - lvalue - to - pointer - expr
Add a test for < rdar : / / problem / 18537723 > Swift instance of var created in Generic func gets wrong type .
Merge remote - tracking branch ' origin / master ' into master - rebranch
[ ConstraintSystem ] Remove implementation of operator designated types
ImplicitClosureExpr ( AutoClosureExpr in future ) : store body as BraceStmt
Merge pull request from lorentey / string - literal - test - fixme
Merge remote - tracking branch ' origin / master ' into master - next
CMake : Make sure that we finish building the compiler before trying to
Sema : Check result builder availability
[ benchmark ] B_D iterations = > independent - samples
Merge pull request from jckarter / assoc - type - inference - same - type - candidate
[ ownership - verifier ] Fix a small thinko from a previous refactoring and add a test to be sure I don ' t break it again . . .
[ Import as Member ] Remove broken swift_name for CGPDFPageGetBoxRect
Update apinotes README with the - target option .
docs : add a rationale for lazy functions on sequences and collections
[ Concurrency ] Ban non - escaping closures in @ asyncHandler
Adding documentation to clarify the result of DispatchTime ( uptimeNanoseconds : 0 ) ( )
Set up protocol conformances properly for the default
valref . rst : elaboration on array syntax
Remove unused variables from Android configuration
Merge remote - tracking branch ' origin / master ' into master - next
Merge remote - tracking branch ' origin / master ' into master - rebranch
[ Omit needless words ] The preposition " of " binds to the word preceding it .
Resolves an issue with large loadable types wherein functions types inside classes misbehaved
Exploratory fix to the Linux build .
[ semantic - arc - opts ] Eliminate all dead live ranges .
[ Frontend ] Add missing ` diagnostics ` component to the default translations path
[ ClosureExpr ] Remove hasAppliedFunctionBuilder .
rip out a bunch of special case parsing logic for function arguments now that tuples have a sane syntax .
StdlibUnittest : Fix NSException catching ( broken by SE - 0054 ) .
Merge pull request from ikesyo / serialization - using - over - typedef
add ~ as an operator .
Merge remote - tracking branch ' origin / master ' into master - next
Factor the solving loop that attempts type bindings out of the main solver .
Merge remote - tracking branch ' origin / master ' into master - llvm - swift5 - transition
[ stdlib / private ] [ OSLog ] Add basic support for interpolating floating - point
Merge remote - tracking branch ' origin / master ' into master - next
Merge remote - tracking branch ' origin / master ' into master - next
SIL : Don ' t serialize imported conformances nested in non - public types
Improve diagnostics for selector collisions with @ objc optional requirements .
[ stdlib ] Array API Review : removeAt = > removeAtIndex
Add additional dead store elimination to the compilation pipeline
Merge pull request from jckarter / closure - sil - convention
[ stdlib ] Correct some ReST markup
context cleanup part 1
SimplifyCFG : Ignore expect - intrinsics for dominator - based cond_br optimization .
Merge pull request from gottesmm / pr - 93c0bf13b0385db5c587ec3901a47e3af1892192
[ Constraint solver ] Lazily populate the expression depth / index map .
AST : Remove restriction preventing the base of a member reference from being an existential type
Move import warning silencing logic
AST : CapturedValue is never used as a DenseMap key
[ test / Index / Store ] For a test , replace ' XFAIL : linux ' with ' REQUIRES : objc_interop '
Merge remote - tracking branch ' origin / master ' into master - rebranch
Update validation test for diagnostic drift .
IRGen : Properly communicate ownership to IRGen from the LoadableByAddress pass
[ Demangler ] Teach node printer to use LabelList for entities
Merge pull request from devincoughlin / exclusivity - static - end - to - end - tests
ASTPrinter : print ' throws ' for generic and polymorphic function types
Don ' t crash when using ` @ unknown default ` with a non - enum type .
Merge pull request from practicalswift / gardening - 20170214
Simplify TypeChecker : : fixAbstractFunctionNames and eliminate a silly crash .
Merge remote - tracking branch ' origin / master ' into master - next
Merge pull request from davezarzycki / irgen_missing_header
Merge pull request from gkaimakas / patch - 1
runtime : move the definition of ` report_bad_alloc_err `
Merge pull request from ahoppen / 01 - record - reused - node - ids
[ docs ] Remove CInteroperability . rst .
Sema : Replace getAccessor ( ) calls in SPI attribute checking
This patch makes it easier for the debugger to
Merge pull request from apple / paste - lldb - rules
Merge remote - tracking branch ' origin / master ' into master - next
Merge remote - tracking branch ' origin / master ' into master - rebranch
Sort includes .
Merge pull request from rudkx / update - implements - test
Merge pull request from stephentyrone / SR - 2782 - truncatingRemainder - doc
Merge pull request from owenv / verifier - consumer
Merge pull request from apple / skip - bench - pending - set - on - fire
[ semantic - arc ] Split load [ copy ] - > load_borrow opt into its own file .
[ SE - 0160 ] Allow @ objc on extensions of classes .
additional changes for the validation testsuite , that I missed
Fix typo : commutitively → commutatively
SR - 11889 : Fixed code review issues
build : fix a typo resulting in an invalid variable
Merge remote - tracking branch ' origin / master ' into master - next
Dump BindOverload constaints more verbosely .
Merge remote - tracking branch ' origin / master ' into master - next
Fix typo : declation → declaration
Merge remote - tracking branch ' origin / master ' into master - next
Ignore flake8 warning W504 line break after binary operator .
[ sil - bug - reducer ] Separate construction of pass pipelines into separate functions .
Enable devirtualization even a method to be invoked is implemented by one of the superclasses .
[ stdlib ] SR - 7266 Make sure to actually run the tests
Update flag name to - debug - info - store - invocation .
Merge remote - tracking branch ' origin / master ' into master - next
Use $ ( cmd ) instead of discouraged ` cmd ` .
Rename getFiles to getFilesToEmit
Fixed typo .
[ SourceKit ] Add test case for crash triggered in swift : : TypeBase : : getMemberSubstitutions ( swift : : DeclContext * )
add some testcases from radars fixed by recent work , NFC .
[ sil - combine ] Fold checked_cast_addr_br when casting into protocol types if informations about conformances is available at compile - time .
[ Clang importer ] Be robust against invalid Objective - C categories .
Merge remote - tracking branch ' origin / master ' into master - next
remove some unnecessary code that produces unfortunate effects when other changes go in .
Merge pull request from practicalswift / gardening - 20161130
[ Mangling ] Define " related entity " operators ' LA ' . . . ' LJ '
Merge remote - tracking branch ' origin / master ' into master - next
[ benchmark ] [ Gardening ] Literate testing ( markdown )
[ utils / build - script ] Pass - - xctest override path to SPM build , when building xctest .
ClangImporter : import size_t as Int
Merge pull request from rudkx / fix - rdar30271695
build : prevent accidental addition of LINK_LIBRARIES
Merge remote - tracking branch ' origin / master ' into master - next
Merge pull request from atrick / silval
[ stdlib - code - size ] Do not produce useless partial specializations for the shift operations on integers
Remove UnqualifiedLookup
Merge pull request from jrose - apple / fail - hello - ill - met
[ NFC ] Only Register Primaries As Dependency Sources
Merge pull request from gribozavr / rm - dead - code
Fix signature of observeValueForKeyPath ( _ : ofObject : change : context : ) to match SDK changes .
Merge remote - tracking branch ' origin / master ' into master - next
[ ModuleInterfaces ] Warn emitting without - enable - library - evolution
Add an assortment of new " always - on " metrics .
Merge remote - tracking branch ' origin / master ' into master - next
Provide a better , tailored diagnostic for result - type mismatches . ( rdar : / / problem / 19800727 )
SILGen : Caller - side codegen for invoking foreign async functions
ObjectOutliner : handle set_deallocating instruction correctly .
Add importing cxx function decls and the basic requisite AbstractionPattern support .
Merge pull request from davezarzycki / diag_qoi_SelfAccessKind
[ remote - run ] Pass stdout through directly too
Merge pull request from DougGregor / isobjc - without - attr
revert changes to stdlib
introduce a new ModuleType , which will be the type given to ModuleExpr ' s .
LookupVisibleDecls : Don ' t use getReasonForSuper ( ) for members of protocols and superclass constraints on an archetype
[ SILOptimizer ] Don ' t optimize casts to protocols with a conditional conformance .
Fix tests
Code cleanups based @ slavapestov ' s review
[ Test ] Un - XFAIL ParseableInterface / verify_all_overlays . py .
introduce ImplicitThisTupleElementExpr , progress towards 10682135 .
Use an ugly series of builtins to eliminate swift_makeUnsafeNil ( ) .
[ ComponentStep ] NFC : Remove redundant ` numComponents ` check for debug output
Merge pull request from benlangmuir / crash - cmake - less
Revert r4125 ; it ' s not going where I want it to .
Merge remote - tracking branch ' origin / master ' into master - next
Enhance the general parameter list matching logic in CSDiags to know about
Fix previous patch .
Merge remote - tracking branch ' origin / master ' into master - next
[ benchmarks ] Add a benchmark for printing using mirrors ( )
Runtime : Use ' once ' instead of static local variable initialization .
Fully embrace ParamDecls instead of AnyPattern at function creation time , thus removing the need to create param decls at SILGen or Sema .
AST : Add a DeclContext field to ComponentIdentTypeRepr
stdlib : Don ' t reserve capacity for filter result array .
add some testcases for situations that produce unpleasant errors .
Merge pull request from aschwaighofer / bump_swift_abi_version
Merge remote - tracking branch ' origin / master ' into master - next
Add support for flag in driver_arguments
Merge pull request from gottesmm / pr - 6da93287e2d5c030e12989e4f935266c72a63b91
API Notes : add UIView + a couple of corrections to UIKit audit .
Merge pull request from adrian - prantl / cleanup
[ ConstraintSystem ] Switch availability check to use correct API
Merge remote - tracking branch ' origin / master ' into master - next
Merge remote - tracking branch ' origin / master ' into master - next
Remove object file from test directory
[ build - script ] Define default llvm_install_components .
Merge pull request from gottesmm / plugging_leaks_1
Merge pull request from jckarter / key - path - true - const
[ SIL ] Dump the < no loc > SILLocation , which means the SILLocation is null .
SILGen : Refactor emitApplyAllocatingInitializer ( ) to take PreparedArguments
Change ABI checker expected file , allowing us to group entries by change , avoiding merge conflicts
AST : Remove a few utility methods from AbstractStorageDecl
Add - global - opt pass .
IDE : Stop using Expr : : getLValueAccessKind ( )
stdlib : remove casts that are not required
Remove Cascading Verification from the Dependency Verifier `
Revert " [ Dependency Scanner ] Add missing clang overlay dependencies for placeholder modules "
Don ' t consider invalid enum entries when looking up enum cases by name ;
SILGen : Fix invalid test case
[ utils / gyb_syntax_support ] Add the classification kind for identifiers
Merge pull request from eeckstein / fix - global_addr
Revert " [ Parse ] Implementation for SE - 200 ( raw strings ) "
Rename isInMainMode to isInputSwift
Merge pull request from bitjammer / sr - 4044 - function - call - expr - syntax
Merge pull request from DougGregor / importer - no - get - members
Always return a valid zone from [ SwiftObject zone ] .
[ csvcolumn_to_scurve ] Avoid division by zero .
IRGen : For classes with resiliently - sized fields , calculate both a fragile and resilient layout
Merge pull request from lorentey / disable - long - running - tests
random comment cleanups .
SIL : Add DeallocRefInst .
SIL : Add devirtualizer support for default witness methods
Merge pull request from swiftix / wip - warn - about - unused - generic - params
Use getSuperclassForDecl
Add a BeginBorrowInst : : getEndBorrows helper .
[ CSRanking ] Change ranking to weight overload choices in evaluation order
Make DeclContext dump ( ) ’ able from the debugger . Unmark Decl ’ s dump method as “ Deprecated and only used by the debugger ” . This isn ’ t doing anything useful since the implementation is used by live code anyway .
Merge pull request from jckarter / subscript - nonnull - crash
Move fine - grained NodeKind to ReferenceDependencyKeys . h
Update UIKit
Merge remote - tracking branch ' origin / main ' into next
Merge pull request from AnthonyLatsis / proto - stubs - nonmutating - set
Merge pull request from eeckstein / fix - arccm
Handle flaky SourceKit / CursorInfo / use - swift - source - info . swift
Merge remote - tracking branch ' origin / master ' into master - next
Merge remote - tracking branch ' origin / master ' into master - next
Merge pull request from atrick / verify - access
[ build ] Evil hack to make Make make
SIL - Combine : fix test case failure on watch
MandatoryCombine : fix an uninitialized variable
[ build - presets ] Don ' t build SwiftSyntax on incremental bots
[ codemotion ] Refactor loop into std : : none_of .
Sema : Diagnose overrides of an @ objc declaration that cannot be @ objc
Enable GenericSpecializer on OSSA by default ( )
Merge pull request from shajrawi / InjectOptional
Fix typos .
Merge pull request from aschwaighofer / irgen_test_case_disable_abi_v7k
[ AST ] Fix a crash when conformance lookup fails during substitution .
Merge remote - tracking branch ' origin / master ' into master - next
stdlib / String : change UTF - 16 / UTF - 16 transcoder in UTF16View to replace
[ Test ] Add some more tests around edge cases
Fix mis - merge
Fix : < rdar : / / problem / 17462274 > Is increment in for loop optional ?
[ string ] Drop many @ inlinable from big API .
[ Parse ] Perform the single expression function body transform for delayed parsing as well
Change TypeBase : : getTypeVariables to return its results in a SmallVectorImpl
Merge pull request from rjmccall / begin_end_access
Sema : Ensure archetypes don ' t leak through when a local generic function uses a typealias defined in function scope
SIL : Remove EnableTypedBoxes staging option .
[ stdlib ] Bidirectional UTF - 8 Prototype : tweaks / cleanups
Merge remote - tracking branch ' origin / main ' into next
Stop using islower ( )
[ AutoDiff upstream ] Add differentiable function type lowering . ( )
Merge remote - tracking branch ' origin / master ' into master - next
Check operator declaration binding in the first pass for FuncDecls .
USR generation : generate USRs for imported macros
Ran my changes through git - clang - format .
Merge pull request from apple / revert - 4776 - needless - warning
IRGen : Make type ( of : ) behavior consistent in ObjC bridged contexts .
Stop running LiterateTests as part of ALL
libSyntax : create syntax nodes for closure signature . ( )
Merge pull request from zisko / master
[ SIL ] Fix use - after - free in ` SILFunction : : print ` . ( )
Merge pull request from adrian - prantl / filter
[ stdlib ] Check in WIP on Strings State of the Union
Merge pull request from gottesmm / pr - 87f11264e3a0d398e96083f801b5ca5c4d3299e5
Actually prefer the representative type variable with the lower ID
Merge pull request from AnthonyLatsis / subscript - fixits - amendments
Cmake changes for - llog and - latomic III
Merge pull request from compnerd / remoteast
Remove the number field
Add FreeBSD to the list of platforms with phreads
Merge pull request from jrose - apple / keep - properties - and - accessors - on - the - same - page
Improve @ guaranteed args handling in ARCSequenceOpts
Merge pull request from hughbe / new - platforms
SILGen : Stub out support for invoking foreign async methods .
Merge remote - tracking branch ' origin / master ' into master - next
[ benchmark ] DataAccessBytes refactored
Revert " Import ' NSError * * ' in parameters as typealias ' NSErrorResult ' . "
[ build - script ] Use a regex to match all the swift tests .
Merge pull request from slavapestov / preserve - sil - when - merging - modules
Preliminary support for OpenBSD in the stdlib .
Make default construction of ' Instruction ' private .
Reorder and remove some passes to reduce compile time .
Made the helper into static function and addressed cosmetic changes .
Merge pull request from practicalswift / typo - fixes - 20160621
Do not penalize optional - to - optional conversions when processing the associated value - to - optional conversions . Doing so can result in incorrect ambiguity errors when applying optional - typed arguments to function overload sets with optional and non - optional parameter types . ( Hitlisted : rdar : / / problem / 18028526 )
Merge pull request from aschwaighofer / remove_escaped_on_convert_escape_to_noescape
[ test ] Switch a test from NSLog to fputs .
Merge remote - tracking branch ' origin / master ' into master - next
Merge remote - tracking branch ' origin / master ' into master - next
[ ParseableInterface ] Clarify call to print accessors for single - entry pattern bindings
Merge remote - tracking branch ' origin / master ' into master - next
Merge pull request from lancep / ArrayNewTestFailures
[ SIL ] Switch InitBlockStorageHeaderInst over to SubstitutionMap .
IRGen : Give ClangCodeGen a chance to emit its translation unit ' s global state
Merge pull request from gottesmm / pr - 413b415c765eb62debe5d09415f22e89952479af
Merge pull request from mikeash / protocol - conformance - iteration - order - workaround
Merge pull request from slavapestov / migrate - noreturn - to - never
Merge remote - tracking branch ' origin / master ' into master - next
Restore updated FloatingPoint init documentation
[ Exclusivity ] Change the exclusivity checks inline heuristic threshold
Merge pull request from phausler / back_port_foundation_tests
Re - enable ARC for RC4 .
Merge remote - tracking branch ' origin / master ' into master - next
Merge pull request from dcci / valuewittarget
[ overlay ] Fix PictureInPicture method names with apinotes ( )
Helper function in ASTContext to get the Array ’ s append element function .
Merge remote - tracking branch ' origin / master ' into master - next
[ stdlib ] OptionSets - final pre - API - review tweaks
demangler : put the demangler into a separate library
ensure that extension members get the proper function type , and a ' this ' argument .
Merge pull request from tfiala / run - lldb - tests - during - pkg
Merge remote - tracking branch ' origin / master ' into master - next
Switch typeCheckArrayBound ( ) over to a listener on top of typeCheckExpression ( ) .
IRGen : Add a counter for GOT entries emitted
[ swiftc ( 44 vs . 5390 ) ] Add crasher in swift : : constraints : : ConstraintGraph : : lookupNode ( . . . )
IRGen : Reabstract lowered dependent types when substituting into a generic context .
Use StdlibUnittest
Merge remote - tracking branch ' origin / master ' into master - next
During coercion , treat expression of generic function type as being
Merge pull request from KevinPayravi / patch - 1
Merge pull request from DougGregor / doc - request - evaluator
Reinstate macosx . cmake because the B & I build script uses it .
[ AST ] Store the TypeReprs of generic arguments used to specialize a decl reference in DeclRefExpr , to impove source fidelity .
[ benchmark ] Gardening : Timer Parasite Control
Merge pull request from adrian - prantl / master - next - compiler - fixes
Merge remote - tracking branch ' origin / master ' into master - next
[ Serialization ] Add a " nested types " lookup table for partial modules .
Merge pull request from rajbarik / raj - cta
[ Type checker ] Experimental support for one - way parameter constraints .
[ build - script ] Don ' t override the LLDB codesign idenity .
Merge remote - tracking branch ' origin / master ' into master - next
Reapply Add a pass to propagate constant array values to array subscript calls
[ benchmark ] [ Gardening ] ` processArguments ` is init
Merge pull request from vedantk / fuzzdebug
Merge remote - tracking branch ' origin / master ' into master - next
WIP unfmt
SIL : Rename ' archetype_method ' to ' witness_method ' .
Fix swapped overload kind checks in solution comparison .
Code completion : if the postfix expression is located on a different line than
[ stdlib ] Generalize array bridge / cast machinery
[ const - prop ] Refactor out worklist initialization into its own function . NFC .
Remove | | closures .
Add a benchmark for protocol conformance testing .
[ devirtualization ] Refactor optimizeClassMethodInst so that the main while loop is smaller and we handle the success case outside of the for loop .
[ SILGen ] Initialization of instance properties with property delegates
12641063 Fix the double - indent problem with properties
AST : Update TypeBase : : getSuperclassForDecl ( ) for subclass existentials
Merge remote - tracking branch ' origin / main ' into next
[ swiftc ] Add 💥 case ( 😢 → 51 , 😀 → 5091 ) triggered in swift : : ArchetypeBuilder : : getGenericSignature ( … )
Re - enable test failing due to the ( now - removed ) - F / S / L / PrivateFrameworks
Merge pull request from eeckstein / fix - temprvalueopt
Merge remote - tracking branch ' origin / master ' into master - next
[ build - script ] Clear CMakeCache . txt before ( re ) configuration .
Merge pull request from swiftix / wip - sr - 1026
move trailing closure tests into the expr / closures directory instead of being
Implement ' if let ' and ' while let ' statements .
[ SIL ] NFC : Repack misc DeallocRefInst bits
Re - enable coverage_smoke . swift to test bots
[ TBDGen ] Class witness table offsets .
Merge pull request from hborla / join - perf - regression
Factor isFinishedWithLookupNowThatIsAboutToLookForOuterResults
rename lexer methods to follow the naming convention and add the ability to lookahead one token .
Merge pull request from aschwaighofer / disable_conditional_conformances_smoke_optimized
Re - apply " [ CodeComplete ] Make sure the lookup API can find closure parameters . rdar : / / 17968255 "
[ stdlib ] Don ' t let gyb interfere with doc comments
[ Parser ] Minor location tweak to aid the scope map .
Debug info : use the new getOrCreateDesugaredType ( ) function to create aliased
DI for TLC seems to have stuck , remove the conditional logic it was hiding behind .
Merge pull request from slavapestov / private - lookup - substitution
REPL : Find completions from Clang modules .
Remove external uses of TypeChecker ' s Context field
Merge pull request from slavapestov / decl - checker - cleanup - part - 6
Merge pull request from nkcsgexi / request - usr - gen
Merge remote - tracking branch ' origin / master ' into master - next
[ SIL ] Add test case for crash triggered in swift : : GenericParamList : : getAsGenericSignatureElements ( swift : : ASTContext & , llvm : : DenseMap < swift : : ArchetypeType * , swift : : Type , llvm : : DenseMapInfo < swift : : ArchetypeType * > , llvm : : detail : : DenseMapPair < swift : : ArchetypeType * , swift : : Type > > & , llvm : : SmallVectorImpl < swift : : GenericTypeParamType * > & , llvm : : SmallVectorImpl < swift : : Requirement > & ) const
Merge remote - tracking branch ' origin / master ' into master - next
Merge pull request from compnerd / windows - runtime
Merge pull request from gottesmm / pr - 72ad962e024d65f236dcbae49c7a0eb2c64c7f83
[ SILParser ] work around PolymorphicFunctionType with input or result type
[ gardening ] 0 = > nullptr . NFC .
[ sil - devirtualizer ] Fix a verifier crash when a generic callee gets devirtualized and replaced by non - generic callee .
Merge remote - tracking branch ' origin / master ' into master - llvm - swift5 - transition
Merge remote - tracking branch ' origin / master ' into master - next
Merge remote - tracking branch ' origin / master ' into master - rebranch
Merge remote - tracking branch ' origin / master ' into master - next
Merge pull request from DougGregor / explicit - availability - macos - only
UInt needs a Mirror of its own , it was simply forgotten when writing the gyb file
Wire up driver support for - serialized - diagnostics and make the consumer implementation private .
[ arc - analysis ] Add a helper routine getFinalReleasesForValue that determines for a value a set of post - dominating live releases .
Debug Info / foreach loops : Use the NamedPattern name instead of " $ generator " when possible .
Merge pull request from troughton / windows - toolchain - pr
SimplifyCFG : fix a crash in the try_apply ( convert_function ) to apply conversion .
ExplicitModuleLoader : use a temporary source manager for loading the explicit module map , NFC
rename closure . swift - > autoclosure . swift
Fix compilation issues with ToT Clang
Merge remote - tracking branch ' origin / master ' into master - next
stdlib : split the Process class into a POSIX - dependent ( EnvironmentVariables )
IRGen : Convert @ objc enums to the Clang types of their raw types .
Replace use of LLVM_IS_FINAL with std : : is_final
Structure for IR generation .
Merge pull request from eeckstein / reinstate - inliner - change
test : mark linker directives tests as macosx only
[ test ] Mark this recently - added crasher as requiring assertions . ( )
[ ConstraintSystem ] Record trailing choice match choice when arguments / result are equivalent to applied function
Merge pull request from rjmccall / no - need - for - bridging - autoreleasepools
fix the " super . init must be called before returning nil " note to also
StdlibUnittest : allow to feed custom stdin input into a test
Merge pull request from apple / expression - breakup
Include module name in string
IRGen : Avoid using the GOT for public symbols in the same TU .
Change CFGValue from being a PointerUnion < Instruction * , BBArg * > to being a base
build : improve libedit handling for builds
Merge remote - tracking branch ' origin / master ' into master - next
[ ConstraintSystem ] Specialize diagnostic for ` ~ = ` to talk about expression pattern use
Merge remote - tracking branch ' upstream / master ' into HEAD
Fix a few ReST issues .
[ Parser ] Support " < " unary operator in # if swift ( ) expressions
Remove some gratutious uses of GCC extensions from the Syntax library .
[ Parser ] Some refactoring inside parseSelectorArgument ( ) , no functionality change .
CODE_OWNERS : Assign Max to all other overlays
[ IDE ] Update the character source ranges for structure nodes so they properly span the entire end token .
Merge pull request from xedin / anyobject - conversion - diagnostics
Add tests for < rdar : / / problem / 19831698 > Incorrect ' as ' fixits offered for invalid literal expressions
[ SE - 0112 ] Bridge NSError to ErrorProtocol .
Reflection : Add - swift - stdlib - enable - reflection - metadata flag for testing
[ Type checker ] Allow the ‘ override ’ keyword on protocol members .
tests : use more precise StdlibUnittest assertions
SR - 11902 : Fixed code review issues . Updated tests .
IRGen : Fix broken destructors for heap arrays .
[ CSGen ] [ CSDiag ] Update SanitizeExpr to santize OpenExistentialExpr
[ libSyntax ] Represent raw string delimiters
Merge remote - tracking branch ' origin / master ' into master - next
Add a regression test for SR - 8968
[ serialization ] Separate extensions out into their own table .
Slow module access down with pointless abstraction .
Merge pull request from shajrawi / outline_copyddr
[ Parse ] getLineNumber before parsing the closure ( )
Revert deletion of comment .
SwiftShims : remove C + + code and unprefixed names from RuntimeShims . h
[ ownership - verifier ] Teach the ownership verifier how to handle unreachable code .
Merge pull request from airspeedswift / test - contiguous - storage
For now turn off computation of loop region analysis while I investigate an assertion failure .
Merge pull request from bob - wilson / pr16362 - fix - followup
add an accessor , NFC .
Merge pull request from phausler / dictionary_bridge_perf
[ test ] Update . cxx_ { construct , destruct } type signature length
Merge remote - tracking branch ' origin / master ' into master - next
Reflection : Extract new decodeMangledType ( ) for when we have a TypeRefBuilder but not ReflectionContext , NFC
Merge pull request from ArtSabintsev / patch - 1
[ Syntax ] Fix type syntax tests for ' async ' .
Parser : Allow static vars in DisallowStoredVar contexts .
SILGen : Tweak key path computed property lowering to produce a consistent runtime - callable ABI .
Fixed grammar mistake
Revert " [ gardening ] Remove unused code . "
Merge pull request from jckarter / lazy - interface - type
Merge pull request from vedantk / cross - compile - compiler - rt
AST : Check invariants in GenericSignature : : getSubstitutionMap ( SubstitutionList )
Fix deadlock in leaks runner script .
[ GSB ] Only use matching superclass constraints in enumerated requirements .
Merge remote - tracking branch ' origin / master ' into master - next
[ driver ] Pass - Xfrontend options to the module - merge tool .
Silence a couple release build warnings about unused variables .
[ driver ] A handful of fixes to make - v only display version info if no inputs are provided .
_fastEnumerationStorageMutationsTarget exists to be usableFromInline
Fix typo : weirdndess → weirdness
introduce a CFGVisitor class , switch the instruction printing logic
fix a comment that John pointed out
AllocBoxToStack : Relax assertion to allow typed boxes .
removing commented out default implementation
Merge pull request from CodaFi / contextual - clues
Merge remote - tracking branch ' origin / master ' into master - next
Merge pull request from varungandhi - apple / vg - consistent - note - style
Merge pull request from gottesmm / pr - 1bf9b6a4d0027fd10e89ae3bb871905923f1748a
Add a - disable - sil - partial - apply frontend staging option .
Merge remote - tracking branch ' origin / master ' into master - next
Merge pull request from ravikandhadai / oslog - strings
LoopRotate : Update comment to reflect reality
Merge pull request from gottesmm / pr - 5d7a1ea2d4ae11d647fceb405a4c2307150f7b06
[ CSFix ] Convert missing explicit @ escaping fix to be a contextual mismatch
Merge remote - tracking branch ' origin / master ' into master - next
Check stderr when looking for incremental remarks .
[ ModuleInterface ] Short - circuit module loading for frameworks too ( )
Strict keyword arguments : fix locator for conversions .
IRGen : Emit category protocols for the conformances of the extension , not for the original type decl .
[ CG Import as Member ] Fix argument label swap
runtime : make LLP64 clean
stdlib : add test case which checks that no trap instructions are generated for a Dictionary lookup
Merge pull request from practicalswift / fix - accidental - double - spaces
Merge remote - tracking branch ' origin / master ' into master - next
XFAIL a crasher that is periodically hanging the compiler in CI .
Merge pull request from apple / add - flag - non - executable - test
Merge pull request from nkcsgexi / extension - from - other
Mark conformances as " used " when the GenericSignatureBuilder needs them .
Merge remote - tracking branch ' origin / master ' into master - next
25847 - emitsimpleassignment . swift crashes with - emit - silgen
Collapse ConstraintSystem : : SharedState ; it ' s all shared now .
Merge pull request from eeckstein / dse
Merge pull request from practicalswift / self
Merge remote - tracking branch ' origin / master ' into master - next
Merge pull request from moiseev / characterview - warning
[ ASTPrinter ] Fix raw string enum cases with unicode scalars
Merge pull request from harlanhaskins / absolute - position - positions - absolutely
[ test ] Fix three Driver tests under cross - compilation . ( )
Revert inaccurate typo fix
Introduce a new Cleanup called EndBorrowCleanup
Merge remote - tracking branch ' origin / master ' into master - next
[ index ] Make sure that we index implicit constructors
[ ClangImporter ] Special - case some known non - CF types whose names end in " Ref " .
build : build the section_magic libraries for ELFish targets
Fix typos in declarations of swift_ptrauth_key_associated_type and swift_ptrauth_key_associated_conformance ( )
Fix a few assertions that were too tight .
[ caller - analysis ] Reimplement CallerAnalysis ontop of findLocalApplySites .
IRGen : Concrete type metadata does not have generic parameters or requirements
[ swiftc ( 30 vs . 5544 ) ] Add crasher in swift : : GenericSignature : : getConformanceAccessPath
[ gardening ] PEP8 : Fix remaining " missing whitespace around arithmetic operator " violation .
Merge remote - tracking branch ' origin / master ' into master - next
Merge pull request from xwu / floating - point - constants
Merge pull request from davidungar / unit - test - users - v5a
Zap unused TypeChecker : : getDefaultLiteralExpr ( Expr * )
Merge remote - tracking branch ' origin / main ' into next
Merge remote - tracking branch ' origin / master ' into master - next
Merge pull request from brettkoonce / 20 . 04 - readme - package
SIL : Add the ability for partial_apply instructions to control the ownership of the produced closure .
Break up AccessControlChecker : : visitPatternBindingDecl a bit
[ CodeCompletion ] Add ' IsSystem ' flag to code completion result item
[ DefiniteInitialization ] Check whether globals captured by top - level
Update tests for accessibility modifiers becoming context - sensitive keywords .
[ stdlib / Foundation ] Swift 3 backward compatibility hack
New indexing model : fix the BridgeNonVerbatim . swift test
[ Property Wrappers ] Make sure captures are computed for synthesized property
[ Foundation ] Add numerous tests for closer to full coverage of IndexPath
Don ' t over - eagerly specialize the return type of an enum initializer application . Doing so doesn ' t really improve compiler throughput , but can cause problems if the enum has overloaded initializers . ( rdar : / / problem / 19697319 )
Merge remote - tracking branch ' origin / master ' into master - next
migrator : ensure we update unresolved member access correctly to global names . rdar : / / 41658300
[ Trailing closures ] Enable " fuzzy " heuristic by default everywhere .
stdlib / comments : UTF - 8 and UTF - 16 are spelled with a dash
Add benchmark target when BUILD_SWIFT_PERF_TESTSUITE is set
Remove extraneous semicolons .
Add more entrypoints to CallGraph to hide implementation details .
build : convert paths to CMake paths before use
Merge remote - tracking branch ' origin / master ' into master - next
Enable stdlib resilience
Merge pull request from jckarter / objc - generic - protocol - conformance
Merge remote - tracking branch ' origin / master ' into master - next
Merge remote - tracking branch ' origin / master ' into master - next
Fix warning in swift - api - digester
Assert that an added file loaded without error
Merge pull request from rintaro / build - script - tweak - args
[ Concurrency ] Require references to ' async let ' to have an ' await ' .
Revert " Revert " Revert " swift - lang : shuffle the source directory structure ( N … "
AST : switch to ` std : : sort ` from ` llvm : : array_pod_sort `
Merge pull request from slavapestov / class - resilience - part - 11
[ Scope map / parser / AST ] Miscellaneous cleanups to avoid producing invalid source ranges .
SILGen : Implement partial application of generic methods .
Fix memberCount for lazy member parsing
Merge pull request from swiftix / conditional - bridging - downcast - bailout
[ AutoDiff ] NFC : Garden test . ( )
Centralize our addition of @ objc attributes in the Clang importer .
fixed : specialization of reabstraction thunk in CapturePropagation results in broken code .
Merge pull request from practicalswift / swiftc - 28608 - swift - typebase - getcanonicaltype
[ RemoteMirrors ] Use RTLD_LOCAL when loading the remote mirrors dylibs in the test program to avoid symbol clashes when loading two of them .
[ sil - mode ] Add highlighting for unconditional_checked_cast_addr .
Everyone loves a good Mirror , and this commit has plenty of them !
libSyntax : some refactoring on syntax parsing context . NFC
Provide a nice little convenience method for testing whether an
Split trySpecializeApplyOfGeneric ( ) from specializeApplyInstGroup ( ) .
[ Definite initialization ] Avoid performing DI via nonmutating setters .
Merge pull request from aschwaighofer / fix_test_prototypes_algorithms
update line - directive for # sourceLocation .
Merge remote - tracking branch ' origin / master ' into master - next
Merge pull request from xedin / rdar - 24329052
Merge pull request from ahoppen / sil - escape - keywords
Merge remote - tracking branch ' origin / master ' into master - rebranch
Merge pull request from shajrawi / builder_asserts
Merge remote - tracking branch ' origin / master ' into master - next
Remove dead parameters from helper functions .
Merge remote - tracking branch ' origin / master ' into master - next
[ ConstraintSystem ] NFC : Unify type variable and disjunction choice representation
Merge pull request from hborla / property - wrapper - default - init
Merge pull request from nathawes / doc - info - missing - submodule - decls
[ Serialization ] Track whether a cross - reference came from Clang ( )
Merge pull request from practicalswift / typo - fixes - 20160207b
[ SR - 4347 ] Improve inference of optional supertypes
address more reviewed issues
Look through reference storage types when generating
< rdar : / / problem / 27425997 > NSCoder encodeInt : forKey cannot be overridden in Swift , throws runtime exception
[ Clang importer ] Remove dead code .
IRGen : Clear up dtor special cases in SIL - IRGen .
Merge pull request from rjmccall / invalid - type - context - in - super - ref
Merge pull request from modocache / quote - driver - test
Merge pull request from xedin / cgr - refactor
move SILFunction : : dump out of line , so that LLDB can find it .
Add doxygen comment .
Sema : We don ' t have to explicitly set the type of the main function
[ ownership ] When erroring since we visited an instruction that we should never visit , dump the instruction .
Remove unused bit in CapturingExpr
Collection . count = > . length
[ IDE ] Don ' t walk to ExplicitCast expr twice in ModelASTWalker
Initialize the DebugInfo field in the default constructor of Options .
IRGen : Only use type layouts when we expect the generated IR to be faster
Merge remote - tracking branch ' origin / master ' into master - next
[ stdlib ] public init for EmptyGenerator
Fixed typo in comments
demangler : Don ’ t emit an empty string with “ with suffix . . . ” if the full name could not be demangled
Rename ConsumerPair - > Subconsumer
Merge remote - tracking branch ' origin / master ' into master - next
stdlib : Fix resilient build
UnXFAIL and fix the array_contentof_opt test
SIL : Remove redundant utility method and rename another one
[ stdlib ] Add _SequenceWrapperType and _CollectionWrapperType
Merge pull request from ravikandhadai / constexpr - checked - casts
[ CSBindings ] Cleanup ` tryTypeVariableBindings `
Merge pull request from lancep / SliceBufferInlinableAudit
DependenciesScanner : include compiled module candidates for textual module interface in JSON output
[ Syntax ] Abolish ' backtick ' trivia
Merge remote - tracking branch ' origin / master ' into master - next
Revert " SIL : Add a project_box insn to project the address of the value from a box . "
follow the predominant pattern of making const ivars const and
Merge remote - tracking branch ' origin / master ' into master - next
[ CMake ] Don ' t pass - sil - serialize - all under SWIFT_STDLIB_ENABLE_RESILIENCE .
Merge remote - tracking branch ' origin / master ' into master - llvm - swift5 - transition
[ Omit needless words ] Always dump both argument label and parameter name .
[ Sema ] Fix crash when checking for overrides among erroneous declarations .
[ sil ] Complement ValueBase : : getParentBB ( ) with ValueBase : : get { Function , Module } ( ) .
test : disable stdlib tests due to rdar : / / 61347183
Gate the swift - remoteast - test tests on the target matching the host .
Merge remote - tracking branch ' origin / master ' into master - rebranch
unfmt
[ benchmark ] ArrayAppend for substrings .
Merge pull request from slavapestov / sil - link - remove - worklist
Sema : Allow extensions of nested generic types via generic typealiases
Merge remote - tracking branch ' origin / master ' into master - next
Runtime : Static - ize swift_unsafeReflectAny .
Merge remote - tracking branch ' origin / master ' into master - next
[ SIL ] [ SR - 6065 ] Take into account @ objc name when determining method family
Merge pull request from JDevlieghere / swift_dir
Sema : Fix matchTypes ( ) for existential to superclass conversions
[ CS ] NFC : Clarify FullyBound doc comment
[ AST ] Remove old code in emitLetToVarNoteIfSimple ( ) that prevented the fix - it from being applied in some cases
Merge pull request from DougGregor / conditional - conformance - equatable
stop serializing the self decl for ctors and dtors , it is redundant with the
[ stdlib ] Remove a bunch of unused functions from Runtime . swift . gyb
Adjust test case for linux
[ CSSimplify ] Fail when try do simplify a key path application to a key path dynamicMemberLookup
Merge remote - tracking branch ' origin / master ' into master - llvm - swift5 - transition
[ serialization ] Accept @ weak and @ unowned on VarDecls .
Handle Unmanaged ( Retain | Release ) ValueInst as RefCountingInst
stdlib : Portable random function
Diagnose when switching on a non - frozen enum without a catch - all case
Merge pull request from practicalswift / swiftc - 28771 - unreachable - executed - at - swift - include - swift - ast - cantypevisitor - h - 41
self_downcast never deals in address - only types .
split the walkorder enum out to its own header file . This is a silly waste of
Implement support for lookup of members of the current
Merge pull request from davezarzycki / nfc_fix_implicit_fallthrough_warning
Merge pull request from gottesmm / pr - f782df1b8f778cdddc48f8343fb0876f598f9012
Merge pull request from Catfish - Man / stringbridgingspeedups
Merge remote - tracking branch ' origin / master ' into master - next
[ benchmark - dtrace ] Enabling multiprocessing option to speed up gathering data .
build : rename prefix for ` _add_swift_host_library_single `
[ test / TBD ] use " internal " instead of " private " .
AST : Refactor TypeBase : : adjustSuperclassMemberDeclType ( ) to use new SubstitutionMap : : getOverrideSubstitutions ( )
Merge pull request from eeckstein / disable - linux - test
remove stale comment
AST : Fix recursive properties of GenericFunctionType
[ Parser ] Update # os ( . . . ) parse tests for r22720 .
stdlib : hide roundUpToAlignment
Merge pull request from xedin / rdar - 47550715
[ gardening ] LLVM style suggests not indenting large namespace blocks .
merge FailureDiagnosis into GeneralFailureDiagnosis .
Merge remote - tracking branch ' origin / master ' into master - next
SwiftPrivate : fix dependencies on Windows
Merge remote - tracking branch ' origin / master ' into master - rebranch
Fix < rdar : / / problem / 22519983 > QoI : Weird error when failing to infer archetype
Re - apply " Sema : Always synthesize accessors for structs , unless they were imported from Clang "
Fixed some whitespace .
Merge remote - tracking branch ' origin / master ' into master - next
[ cmake ] Fix some errors in the cmake code that enables LTO .
Handle default actors by special - casing layout in IRGen instead
Merge pull request from slavapestov / unbound - generic - type - cleanup
Merge pull request from mikeash / fix - new - error - test - back - deployment
[ AST ] NFC : Switch ` Space ` printing to use ` interleave ` instead of hand rolled loops
[ Sema ] Add warning for ambiguous value assignment when using Optional ( )
Merge remote - tracking branch ' origin / master ' into master - next
Merge pull request from slavapestov / class - method - abstraction
Constify PrintOptions in ShouldPrintChecker : : shouldPrint
stdlib : remove _Reflectable
Merge pull request from slavapestov / dependent - conformance - records
[ Runtime ] Eliminate ProtocolConformanceFlags .
Remove dead linking code from call graph .
Merge remote - tracking branch ' origin / master ' into master - next
Merge pull request from gottesmm / eliminate_deallocating_convention
Merge pull request from DougGregor / sub - map - concrete - generic - params
Update BidirectionalIndexType . advancedBy ( _ : _ : ) for removal of + + operator
Merge pull request from aschwaighofer / revert_array_first
IRGen : Delete dead functions
Merge remote - tracking branch ' origin / master ' into master - next
Merge pull request from apple / run - ios - host - tests
Merge remote - tracking branch ' origin / master ' into master - next
Merge pull request from practicalswift / sourcekit - 071 - swift - printoptions - setarchetypetransform
Replace two # ifs with # if / # else in new integer random algorithm ( )
Replace std : : ofstream with llvm : : raw_fd_ostream in - verify - apply - fixes ( )
Merge pull request from DougGregor / printasobjc - typealias - nontypedecl
IRGen : Fix type of a global with tail allocated storage
AST : Refactor Substitution : : subst ( ) to use GenericParamList : : getSubstitutionMap ( ) , NFC
Merge remote - tracking branch ' origin / master ' into master - next
Merge pull request from apple / revert - 6679 - archetype - anchor - cleanup
Merge pull request from gottesmm / pr - d1020e7d823eb26f201c3e78a39396514590a77e
add void type .
Merge remote - tracking branch ' origin / master ' into master - next
Merge remote - tracking branch ' origin / master ' into master - next
[ ABI ] Don ' t emit overriding associated type declarations into witness tables .
Merge pull request from rudkx / add - test - for - rdar32057712
Merge pull request from benlangmuir / completion - escaping - attr
Add support for generating doxygen compatible external search indices for swift .
Improve the location information for class and value initializer bodies .
Fixes bug in exhaustive matching check SR - 11160 .
Fix a - debug - only = silgen crash .
Merge pull request from valeriyvan / RemoveRedundantBufferZeroingUnicodeScalarProperties
Fixes for typealiases involving generics ( )
[ NamedLazyMemberLoading ] Bail on cases that interact with protocol member mirroring .
Use ProtocolDecl : : getProtocols ( ) rather than : : getInherited ( ) .
stop emitting initialize_var instructions in enum constructors . This is
[ benchmark ] BernchmarkDriver run in batch mode
[ docs ] LibraryEvolution : Fill out " Fixed - Layout Structs " .
[ demo ] Update Obj - C version to match splitting on \ n and \ r .
Merge pull request from Rostepher / append - no - more - stdlib - deployment - targets
Merge pull request from glessard / nsstringapi - curry - fix
LICM : Allow hoisting sideeffect - free apply instructions out of a loop .
Merge pull request from gregomni / tuple - element - lvalue
Resolve fixed crashers .
Test : Added an example with access to a concrete associated type through Self .
IRGen : Start using ExistentialLayout
[ coverage ] Reworked CMake invocation for coverage testing given there are three states for SWIFT_ANALYZE_CODE_COVERAGE
IRGen : Make partial application forwarders honor the convention of the applied parameters .
build - script : reduce duplication of check
[ TBDGen ] Static variables in the main file have accessors .
build : track object format with SDK
Implement support for allocating a new array within the
EscapeAnalysis : fix bug in alias checking regarding ref_element_addr
Merge pull request from CodaFi / east - extern
Merge remote - tracking branch ' origin / master ' into master - next
Merge pull request from slavapestov / sil - parser - astscope
Reapply " [ sil - arc - opts ] Enable single basic block code motion of retain instructions . "
[ semantic - arc - opts ] When performing load [ copy ] - > load_borrow on classes , do not ignore forwarding uses .
Merge pull request from rudkx / expression - too - complex
- ast - dump shouldn ' t crash on empty identifiers
[ stdlib ] made String . utf16count unavailable
Merge remote - tracking branch ' origin / master ' into master - next
Merge pull request from Gankro / roundup
Add changelog entry for optional - to - archetype casting behaviour change
Merge remote - tracking branch ' origin / master ' into master - next
Merge remote - tracking branch ' origin / master ' into master - next
[ utils / build - script - impl ] Pass header / lib paths to the syntax parser library for the build - script invocation of SwiftSyntax
Don ' t emit line table entries for SIL functions marked as thunks .
stdlib / Dictionary : continue removing the assumption that keys and values are
Merge pull request from slavapestov / conformance - access - paths
Fix a scary and long standing bug which never manifested because vars and funcs had the same data .
[ Reflection ] Implement TypeRef demangling for foreign classes .
Serialization : Fix latent bug with extensions of nested generic types
Update the toolchain plist by moving SWIFT_USE_DEVELOPMENT_TOOLCHAIN_RUNTIME into OverrideBuildSettings
Sema : Fix source range for curry thunks
Merge pull request from troughton / patch - 1
SILGen : Only set the external decl of a key path component if the accessor is public
Use the same value for LeastValidPointerValue in the
Address @ jrose - apple ' s code review comments . ( )
put back _sanityCheck
It is considered poor form to store stack - allocated memory in the AST .
[ CMake ] Use NO_CMAKE_FIND_ROOT_PATH for LLVM and clang
Merge pull request from apple / enable - ios - simulator - 32bit - test
[ Runtime ] Reimplement initialization of resilient witness tables .
Merge pull request from briancroom / relative - symlinks
SILGen : Let any reftype expression root an lvalue .
Sema : Fix bug in buildMemberRef ( ) with subclass existentials
Specifically mention version mismatches for non - subclassable classes .
Code completion : eat the code completion token where we handle it
[ silgen ] Use SILGenBuilder APIs when creating value metatypes .
Merge pull request from davidungar / PR - 18 - 15 - num - threads - fix
SIL Serialiation : add Transparent to SILFunction so we can decide which
[ closure - spec ] Teach closure specialization how to handle partial applies with one use .
[ stdlib ] Add _UnitTestArray [ Buffer ] temporarily
SILSerializer : serialize 3 more witness entry types .
Keep tests for rdar : / / problem / 27796375 behavior .
Merge remote - tracking branch ' origin / master ' into master - next
Rename clang : : vfs to llvm : : vfs
tests : fix syntax errors in CHECK lines so that they are actually active
Merge pull request from bnbarham / fix - flakey - vfs - tests
Merge remote - tracking branch ' origin / master ' into master - next
[ CMake ] Stop requiring $ { SDK } / System / Library / Frameworks / module . modulemap
[ swiftc ( 40 vs . 5431 ) ] Add crasher in swift : : TypeBase : : getCanonicalType ( . . . )
Merge pull request from linux - on - ibm - z / s390x - smallstring - v2
Merge remote - tracking branch ' origin / master ' into master - next
Revert " [ stdlib ] _ArrayBufferType : drop baseAddress "
Code completion tests : check declaration kind , it is computed with non - trivial logic now
Merge pull request from apple / revert - 28295 - pr - 5dfaf35e4af5aa2b6e46e3db704bc973cd4e6d53
Merge pull request from compnerd / wintls
Flip the switch on ' private ' and ' fileprivate ' .
Merge remote - tracking branch ' origin / master ' into master - next
Fix recently introduced typo ( " an variable " → " a variable " )
Merge pull request from compnerd / sil - iwyu
[ stdlib ] Improve a doc comment
Use a void * instead of an opaque sourcekitd_uid_t pointer in SKUIDToUIDMap .
Merge pull request from CodaFi / measure - twice - cut - once
Merge pull request from benlangmuir / unsound - cursor
AST : Return abstract conformance for UnresolvedType
Merge remote - tracking branch ' origin / master ' into master - rebranch
[ Diags ] Remove dead code already handled by diagnoseSimpleErrors ( )
[ CSDiag ] NFC : Remove all code from visitInOutExpr since it ' s now obsolete
Merge remote - tracking branch ' origin / master ' into master - next
SILCombiner : Use ASTContext : : getExistentialSignature ( )
[ stdlib ] Added ASCII fast path for string hashing
Exclusivity access marker verification . Handle Unsafe access .
Fix libSyntax support for yields with expression lists .
Merge pull request from jckarter / use - assoc - type - conformance
Re - enable contractions for parameter binding constraints between identical type variables .
Rationalize " throws " as a syntactic choice .
Change oneof - > union in a comment
remove an old spreadsheet that has lost its utility and move another proposal out .
De - intrinsify and remove the _does * OptionalHaveValue functions .
[ func - sig - opts ] Restrict the set of functions that we will optimize .
Merge pull request from aschwaighofer / array_witnesses_runtime_impl
Integrate python linting into validation tests
Changes for code review
Steal files from other branch
Add a TypeBase : : isAny helper .
Fixed bug where SILInstructionVisitor was expecting visitSILArgument to have ValueRetTy not void .
[ Frontend ] Infer simulator environment in the frontend , too .
Misc . cleanup related to use of TranslationUnit .
Merge pull request from slavapestov / remoteast - fixes - part - 1
[ benchmark ] Move build configuration / sdk configuration out of the main CMakeLists . txt file into AddSwiftBenchmarkSuite . cmake .
AST : Fix crash with invalid protocols in SubstitutionMap : : getProtocolSubstitutions ( )
Merge remote - tracking branch ' origin / main ' into next
Merge pull request from atrick / fix
Allow bodiless constructors , destructors , and subscripts when parsing Decls in SIL mode .
Requestify SourceFile parsing
Merge pull request from swiftix / create - apply - simplification
Merge pull request from practicalswift / swiftc - 28813 - swift - genericsignature - enumeratepairedrequirements - llvm - function - ref - bool - swift -
Merge pull request from CodaFi / lassen - fleigen
[ Clang importer ] Remove unused struct CachedExtensions .
Merge remote - tracking branch ' origin / master ' into master - rebranch
[ docs ] [ Windows ] Build clang in Release mode instead of Debug
[ coverage ] Fail gracefully on non - Darwin
notes from doug ' s feedback
Merge remote - tracking branch ' origin / master ' into master - next
Merge remote - tracking branch ' origin / master ' into master - next
Merge pull request from antonblanchard / powerpc64_merge
API Notes : yaml - to - binary : fix nullability mapping inconsistencies .
[ ClangImporter ] Add ClangImporter : : getClangModuleHash ( ) .
Merge remote - tracking branch ' origin / master ' into master - next
Merge pull request from davezarzycki / nfc_fewer_reserved_TypeBase_bits
[ upstream - update ] Work around availability issue .
[ HACK ] Fix r3837 ' s test for 10 . 9 vs . 10 . 8 systems .
Merge pull request from nkcsgexi / mangled - archetype
Merge remote - tracking branch ' origin / master ' into master - next
TBDGen : teach the compiler to take a json file indicating previous install names
Merge pull request from practicalswift / fix - double - spaces
os overlay : Fix XPC dependencies . Hopefully the last change with this
Merge remote - tracking branch ' origin / master ' into master - next
[ Gardening ] DataBenchmarks remove unused string
[ Driver ] Actually set extra environment variables in single - command mode .
Merge pull request from hamishknight / coercion - indirection - changelog
SIL : Set the context generic params of reabstraction thunks less greedily .
swift - reflection - test : Funnel all test output to stdout
libSyntax : create separate node kinds for quote ( " ) and multiline quote ( " " " ) .
[ CMake ] Move swift_llvm_config to SwiftSharedCMakeConfig ( for SourceKit ) .
Include inner class members in dynamic lookup results .
Don ' t run LLDB cross compile host test suite
Sema : Bail out early from checkOverrideAccessControl ( ) if we have protocol requirements
Merge pull request from rjmccall / more - metadata - cache - improvements
Merge remote - tracking branch ' origin / master ' into master - next
SIL : Remove ' getPreLoweredType ' from SILType .
Add is_unique documentation .
Merge pull request from adrian - prantl / scopelocs
Solution application : handle construction of struct types .
Merge pull request from benlangmuir / ordering_x86 - sahf
Merge remote - tracking branch ' origin / master ' into master - next
parseDeclFunc ( ) : sink error recovery for bad signature into
Merge remote - tracking branch ' origin / master ' into master - llvm - swift5 - transition
[ benchmark ] Sprinkle more autoreleasepools
[ ConstraintSystem ] NFC : Remove obsolete code related to fixes from Solution
[ driver ] Switched from - - help - hidden to - help - hidden .
round up size and alignment to a power of two . Occurred to me in the shower .
Create a fast path for not - final release instruction .
Push SILBoxType : : getFieldType into SIL and make it take a SILModule .
[ Syntax ] Parse " if statement " and " guard statement " nodes
[ Sema ] Only warn when exporting implementation - only types in SPI
Merge remote - tracking branch ' origin / master ' into master - next
Remove unneeded reference from template function type .
Remove all dependencies on the stdlib from ARC Sequence Opts tests except for the uniqueness check tests
DeadStoreElimination : don ' t require values stored back to memory if the control flow ends in an unreachable
[ stdlib ] fixed inconsistent syntax in test
Merge pull request from Catfish - Man / going - public
[ QoI ] Say " ' @ objc ' " rather than the redundant " ' @ objc ' attribute " in diagnostics .
Merge pull request from modocache / sr - 2855 - driver - show - incremental - args
[ stdlib ] UnicodeScalarView : RangeReplaceableCollectionType
IRGen : Use stable backreferences when profiling circular archetype references .
[ BatchMode ] Sort batch inputs rather than asserting ; they are not always sorted .
add comments
Merge remote - tracking branch ' origin / master ' into master - llvm - swift5 - transition
Merge remote - tracking branch ' origin / main ' into next
Code completion : don ' t assert if module loading failed
[ WIP ] Fix LLDB . framework install path in build - script .
stdlib / shims : declare CoreFoundation APIs only when we have ObjC interop
Merge remote - tracking branch ' origin / master ' into master - next
Use StringRef in more places to match recent llvm changes .
SourceKit : remove ` LINK_LIBS ` ( NFC )
[ docs ] Initial ( empty ) standard library programmer ' s guide
update comment for " constructor " - > " init " syntax change .
Merge pull request from shajrawi / large_type_bugfix_part1
[ Tools ] Remove Foundation dependency from swiftdt , which gets very complicated for building .
Merge pull request from CodaFi / heading - for - disaster
Move NSObject extensions to the ObjectiveC overlay
Merge pull request from mikeash / conditionally - disable - runtimefunctioncounters
Merge remote - tracking branch ' origin / master ' into master - next
[ gardening ] Use quote style consistently .
Merge remote - tracking branch ' origin / master ' into master - next
stdlib : when linking arclite , link in CoreFoundation
Merge remote - tracking branch ' origin / master ' into master - next
[ erel - matcher ] Do not throw away the list of releases if we do not have a joint post - dominating release set . Instead , use a flag .
build : add Bcrypt . h to Windows SDK overlay
Fix computation of “ favored ” constraints for binary expressions < rdar : / / problem / 17943223 > .
[ DeclChecker ] Avoid setting ErrorType on the enum cases with duplicate names
Check conformances against the requirement signature using module lookup .
Remove wrapper functions that added little value . ( NFC )
Code completion : implement delayed parsing for accessor bodies , allowing code
Add XFAIL : no_asserts to compiler_crashers_2 / 0001 - rdar19792730 . swift
Merge remote - tracking branch ' origin / master ' into master - next
Merge remote - tracking branch ' origin / main ' into next
Merge remote - tracking branch ' origin / master ' into master - next
Merge pull request from DougGregor / gsb - nested - types - lazier
[ AST ] NFC : Tail allocate CollectionExpr elements
[ Sema ] Validate extension via LazyResolver in computeDefaultAccessLevel .
Merge pull request from rintaro / nowitness - stub
Merge pull request from nate - chandler / concurrency / irgen / forward - task - and - executor - through - partial - apply - forwarder
swift - api - digester : detect type alias of raw types change to RawRepresentableStructs .
Merge remote - tracking branch ' origin / master ' into master - next
[ TypeChecker ] Minor : corrected safety check against crash in type checker on invalid input .
update comment and fix 80 - col violation : the StringLiteral expr and
Constraint generation rewrites some already - type - checked expression nodes ; make those rewrites stick .
Merge remote - tracking branch ' origin / master ' into master - rebranch
Check overlap of AST nodes .
Merge pull request from shajrawi / init_opaque
[ Update Checkout ] Print sorted repo list
[ ownership - verifier ] Forwarding instructions only are lifetime ending uses when forwarding owned semantics .
Fix Typos
Add runtime - related cases to StdlibUnittest ' s TestPredicate
[ unittests ] Use ASSERT_EQ
[ SILGen ] Output a different message for failed IUO force - unwraps
Merge pull request from gottesmm / pr - ecca9752729a232b491c1343641f4e81a2a3262a
Sema : Fix multi - file edge case with accessor availability checking
Merge pull request from eeckstein / driver - options
build : remove tensorflow - swift - apis from the master branch
Link swift against libclangToolingCore instead of libclangTooling .
Remove - enable - string - pointer - conversion staging flag .
Merge pull request from weaksauce / master
Associate argument labels with the callee in the solver .
Diagnose redefinition of a binop with a nonbinop and visaversa .
Merge remote - tracking branch ' origin / master ' into master - next
Canonicalize ' release ' instructions and make them amenable to code motion .
[ AST ] Add Type : : wrapInPointer ( PointerTypeKind )
Merge pull request from lancep / 🐮 Tests
[ gardening ] Fixed missing syntax highlighting in changelog
Merge pull request from rintaro / ide - completion - case - rdar35943849
add a " container " version of interleave , use it in SILPrinter a bit more .
Merge remote - tracking branch ' origin / master ' into master - next
Merge remote - tracking branch ' origin / master ' into master - next
omitNeedlessWords : Avoid an extra temporary SmallString
build : ignore non - standard C warnings on Windows
[ Omit needless words ] Only move " lonely of " arguments back to the base name .
Re - enable / update two tests that have been disabled for a long time .
Merge remote - tracking branch ' origin / master ' into master - next
Merge remote - tracking branch ' origin / master ' into master - next
Sema : Diagnose inaccessible implicit property wrapper inits again
Merge remote - tracking branch ' origin / master ' into master - next
Wire up basic parsing support for @ unavailable . WIP .
Merge pull request from apple / revert - 6203 - cs - typemap
Merge pull request from milseman / tls_ftw
Remove an unused macro .
Disallow conforming to ObjC protocols with requirements that can ' t be imported .
Merge remote - tracking branch ' origin / master ' into master - next
Merge pull request from xedin / rdar - 38159133
Merge pull request from graydon / more - stats - tidying
utils : support quoted block names in the viewcfg script
Add a callback to swift : : reflection : : MemoryReader that allows LLDB to provide
Merge pull request from 3405691582 / environ_llvm_assert
Merge remote - tracking branch ' origin / master ' into master - next
Support Optional Injection of Opaque value types
Merge pull request from sarveshtamba / master
IRGen : Emit protocol conformance records .
IDE : cache resolved cursor info in the request evaluator
test : Add parens to print for Python3 compatibility
EscapeAnalysis : remove an unused parameter from canOptimizeArrayUninitializedCall
Merge pull request from slavapestov / fix - outlined - thunks
Merge pull request from jrose - apple / check - the - type - before - you - decl
Merge pull request from compnerd / python - 2 . 7 - is - the - future
s / suffix / prefix / , and explain the choice of T .
[ AST ] Use TinyPtrVector for the list of overridden declarations .
move diagnoseArgumentLabelError to CSDiags . cpp now that it is the only client .
Merge remote - tracking branch ' origin / master ' into master - next
Merge remote - tracking branch ' origin / master ' into master - next
[ gardening ] Fix recently introduced typo : " anyting " → " anything "
Now that memopt is its own SIL pass , split it out to its own file . Now
Flush ImmediateFunctionInput
Merge remote - tracking branch ' origin / master ' into master - next
[ + 0 - all - args ] For staging in purposes , add in a lit feature to distinguish in between tests that must run with a + 0 / + 1 runtime .
[ TypeChecker / SE - 0213 ] NFC : Fix one more improved constant propagation test
IRGen : Lower metatypes respecting SIL ' s ' thin ' bit .
Merge pull request from xymus / print - less - objc - imports
Extend - Womit - needless - words to warn about uses as well .
Mark a recent compiler crasher as non - deterministic
Remove evaluator : : getScopeForAccessLevel
Merge pull request from numist / numist / application - denied
Remove redundant space in first ( where : ) of Sequence ( )
IDE : Fix formatting of closing square brackets
Implement the rest of sema for scoped identifier references ( X : : Y ) we now get a proper DeclRefExpr , and this :
Merge pull request from moiseev / strides
Fix parsing of " / / expected - error @ + 1 2 { { message } } " by the verifier .
Merge remote - tracking branch ' origin / master ' into master - next
Fix issue with switch_enum simplification .
[ docs ] LibraryEvolution : Fill out the section on fixed - contents classes .
[ stdlib ] Generate RandomAccessCollection defaults
Restore array value semantics test for C pointer interop .
[ load - store - opts ] RetainValue is insert from a load store opts perspective despite it having side effects due to it touching reference counts .
Parser : improve recovery when the extension name is a keyword
Factor use of the CodeCompletion sys : : Cache into get ( ) and set ( ) methods
Code completion : completion of type - ident in type contexts ( like Foo . # ^ A ^ # )
Merge remote - tracking branch ' origin / master ' into master - next
Remove SILArgumentConvention : : Direct_Deallocating
Merge pull request from rockbruno / swift - keypath - index
Remove for loop in computing mantissa double value ( )
[ CodeComplete ] Add code completion for Clang sub - modules at import decls . rdar : / / 23244877
Fix release build warning .
Merge remote - tracking branch ' origin / master ' into master - next
test additive mutations too
Merge remote - tracking branch ' origin / master ' into master - next
Implement a ' throw ' instruction in SIL .
buildbot : Make sure module cache is scrubbed .
Merge remote - tracking branch ' origin / master ' into master - next
improve error recovery in { } ' s by using less hacky recovery .
[ RemoteMirrors ] Move ownsObject into ReflectionContext .
Merge pull request from gparker42 / GrP - PR - C6FB8379 - 62A0 - 420A - A05D - FBDFEC51F9A3
Ignore trailing whitespace errors in python - lint
Provide a main executable path to populate the Glibc module map path
SILCloner : Factor out the debug scope cloning into a reusable ScopeCloner .
Merge pull request from compnerd / error - recording
Merge remote - tracking branch ' origin / master ' into master - next
Remove Swift 3 compat code from getForeignRepresentable ( ) .
Add support for LLVM bitcode sections
Merge pull request from apple / rdar - 33307780
[ stdlib ] indexing model : Interval / Range merge
Stop giving local types private linkage ; the debugger may still access them .
Merge remote - tracking branch ' origin / master ' into master - llvm - swift5 - transition
Merge remote - tracking branch ' origin / master ' into master - next
Merge remote - tracking branch ' origin / master ' into master - next
[ swiftc ] Add test case for crash triggered in swift : : ArchetypeBuilder : : addRequirement ( swift : : Requirement const & , swift : : RequirementSource )
Merge pull request from hborla / remove - default - literal - favoring
Merge pull request from rjmccall / get - with - ext - info
swift - api - digester : avoid synthesizing nominal types from the same module .
[ test ] Update autolinking tests to match LLVM .
Merge remote - tracking branch ' origin / master ' into master - next
Initial benchmarking in Swift support
Update switch statements for recent LLVM update . ( )
Merge remote - tracking branch ' origin / master ' into master - next
Merge remote - tracking branch ' origin / master ' into master - llvm - swift5 - transition
infrastructure to record whether an ApplyExpr is implicitly async
[ CodeCompletion ] Complete ' await ' in expression position
Merge pull request from aschwaighofer / fix_abi_v7k_v2
WinSDK : add convenience conversions for FILETIME / time_t
Import Objective - C properties annotated with ' iboutletcollection ' as typed collections .
Split GenFunc . cpp in three : GenFunc for function types and blocks ,
[ stdlib ] Rename Char . swift = > UnicodeScalar . swift
Merge pull request from compnerd / build - is - not - path
Merge remote - tracking branch ' origin / master ' into master - next
[ SR - 8340 ] Improve fix - it for var and subscript in Protocol ( )
AST : Remove FunctionType : : getInput ( )
Merge remote - tracking branch ' origin / master ' into master - next
Merge remote - tracking branch ' origin / master ' into master - next
Revert " [ docs ] DebuggingTheCompiler . rst = > DebuggingTheCompiler . md . "
Merge remote - tracking branch ' origin / master ' into master - next
Merge remote - tracking branch ' origin / master ' into master - next
[ Clang importer ] Import Swift 2 " stubs " to improve errors in " Swift 2 " code .
Merge remote - tracking branch ' origin / master ' into master - next
Merge pull request from Rostepher / layouts - install - with - compiler
Merge pull request from davezarzycki / pr31528
Merge pull request from DougGregor / gsb - bump - ptr - alloc
Merge pull request from DougGregor / conditional - requirement - substitution
Merge pull request from erg / cmake - sourcekit - asan
[ IDE ] Add CodeCompletionResult : : getKind ( ) .
Pending QOI improvements , reinstate implicit bridging conversions .
Merge pull request from ezephir / basic - sil - linux
Merge remote - tracking branch ' origin / master ' into master - next
[ Frontend ] Inline parseAndCheckTypesUpTo
Merge pull request from MaxDesiatov / fix - clang - opencl
[ swiftc ( 133 vs . 5184 ) ] Add crasher in swift : : IterativeTypeChecker : : isSatisfied ( . . . )
Merge remote - tracking branch ' origin / master ' into master - next
AST : Assign interface types to ParamDecls
Merge pull request from gottesmm / pr - 068c882956f6e9746cd2ea87e6aea7d6df8f1ac0
[ ConstraintSystem ] Adjust ` diagnoseAmbiguity ` to use a local slice of overloads
[ benchmark ] Use dot in integer parsing benchmark names
Merge remote - tracking branch ' origin / master ' into master - next
[ ClangImporter ] Update clang invocation to use the marketing names of CPUs
SILGen : Use ` super_method ` for native non - final methods in classes
Merge pull request from CodaFi / unseen - university
Let passes get their options ( current configuration ) from the
Merge remote - tracking branch ' origin / master ' into master - next
Introduce TypeBase : : getTypeOfMember ( ) to produce the type of a member access .
Merge pull request from sryze / master
This test case is crashing again . Acknowledge the regression .
Fix test / TypeDecoder / foreign_types . swift for 32 - bit
Improve type resolver to understand the difference between an " immediate " function
PrintAsObjC : Print @ objc enums .
Merge remote - tracking branch ' origin / master ' into master - next
Reserve a larger range for enumerated metadata kinds .
Make TypeChecker : : preCheckExpression ( ) static .
Sema : use the underlying type when available instead of the structural type request
Fixes example snippets in MutableCollection . swift
Disable an assertion to unbreak code - completion
[ gardening ] Remove unused method isOuterArchetype ( . . . )
s / can not / cannot / g
Begin inferring throwing function types for closures . ( There ' s more work to do here - hence the thin tests - but I need to investigate a couple of sema bugs before moving forward . )
Add an < < operator to pretty - print SILInstruction : : MemoryBehavior .
Merge pull request from slavapestov / func - type - arena - fix
Merge remote - tracking branch ' origin / master ' into master - next
Only ignore private decls at global level .
[ SpaceEngine ] Remove ` isUseful ` due to eager empty sub - space pruning
Update stdlib / private / SwiftPrivate / SwiftPrivate . swift
Parse : Fix crash when alignment is too large
[ proj ] Add support to projection for getting all first level projections of an aggregate type and reforming from that type .
Fix a few VarDecls that are initialized with a PatternBindingDecl but which aren ' t
Merge remote - tracking branch ' origin / main ' into next
Merge remote - tracking branch ' origin / main ' into next
[ + 0 - normal - args ] Make sure to use | = when enabling + 0 in sil - opt .
IRGen : Update lowering of SIL closure insts .
SourceKit : Register type checker request functions
Merge pull request from apple / compnerd - prebuilts
[ stdlib ] Add Indexable protocol and adapters
Merge pull request from atrick / remove - swift3 - benchmark
Merge remote - tracking branch ' origin / master ' into master - next
stdlib : remove a redundant protocol requirement re - declaration
Sema : Remove matchFunctionParamTypes ( )
Merge remote - tracking branch ' origin / master ' into master - llvm - swift5 - transition
XFAIL this testcase , which started failing with the changes for SE - 0109 ,
[ Syntax ] Update Status . md for type nodes
[ Mangler ] Support functions with old parameter label mangling scheme
Merge pull request from xedin / rdar - 42678836
Add a new ApplyExpr expression AST node for function application .
IRGen : Replace a few AST isResilient ( ) queries with IRGenModule : : isResilient ( )
Don ' t import Foundation in comment conversion tests
[ IRGenSIL ] Force early materialization of liverange - extended values
valref . rst : Some words about ObjC interop .
Merge remote - tracking branch ' origin / master ' into master - next
Reinstate Jordan ' s r18233 now that our branches have caught up with the Clang API change .
[ test ] Fix arm64 TBI test to emit assembly again ( rather than IR ) .
Restore other test
rework PatternMatchEmission : : emitSharedCaseBlocks to avoid emitting empty blocks
Merge pull request from practicalswift / sil - 037 - swift - sourcefile - lookupcache - lookupclassmembers
Merge pull request from davidungar / PR - 18 - 3a
Delayed function body parsing : replace two error handling code paths with asserts
fix rdar : / / 16280138 - synthesized getter is defined in terms of archetypes , not interface types
Merge pull request from gottesmm / pr - b72f035964186b8844785fa3c3c0566fff73f814
[ NFC ] Add explicit # include for DiagnosticsSema . h in TypeCheckGeneric . cpp .
Expose ExitableFullExpr scopes to the debug info , so variables declared in
Merge pull request from xedin / rdar - 50909555
Merge remote - tracking branch ' origin / master ' into master - next
[ IRGen ] Added LinkEntities for AsyncFunctionPointer .
[ build ] debug symbols for stdlib in RelWithDebInfo
Sourcekit / Indentation : avoid indenting dot member access appearing after trailing closure
Overlay for - [ NSPredicate predicateWithFormat : ] and
Merge remote - tracking branch ' origin / master ' into master - next
Fix a test on Linux
stdlib : extinguish \ brief and \ c
Merge remote - tracking branch ' origin / master ' into master - next
Add support for the @ final attribute on classes .
SIL : Clean up requiresForeignEntryPoint ( )
Merge pull request from aschwaighofer / assume
Merge pull request from gottesmm / pr - b1a0a45c5f7ddde8879ac6fcf10003e0805603f9
Merge pull request from apple / self - reflection - is - difficult
[ reflection ] Extract out ReflectionInfo from TypeRefBuilder - > ReflectionInfo .
Sema : Fix call to Self - returning methods with refined protocols
test : improve test coverage rate for Windows
Merge remote - tracking branch ' origin / master ' into master - next
Add _isFast predicate that is true at Ofast
[ Constraint solver ] Disable shrink ( ) when constraint propagation is enabled .
add a simple ' make wc ' target , to do a wordcount . Up to 3400 lines .
[ FixCode ] Add @ escaping when overriding mismatch is because of it . ( )
[ llvm - arc - opts ] Remove retainAndReturnThree .
[ AST ] Tail - allocate context types in GenericEnvironment .
Merge remote - tracking branch ' origin / master ' into master - next
[ TBD ] Thread enough through to use UniversalLinkageInfo .
Merge pull request from gottesmm / pr - 593fd7aed5142c9e2b0b2f18bf47074e65a04005
Merge pull request from DougGregor / fewer - fewer - gsbs
Merge pull request from weissi / jw - nio - cp - benchmark
Merge remote - tracking branch ' origin / master ' into master - next
Merge pull request from practicalswift / swiftc - 28292 - swift - valuedecl - settype
Teach sema to synthesize the getter / setter for willSet / didSet properties ,
StdlibUnittest : autorelease stub for the native runtime should call the closure
Merge pull request from EliaCereda / sr - 197
Merge remote - tracking branch ' origin / master ' into master - next
Fix doxygen comment . NFC .
Merge remote - tracking branch ' origin / master ' into master - next
pass compilation in to loadAllInfo in prep to do removed files
[ CMake ] Generalize FreeBSD build rather than hardcoding version ' 10 ' .
[ Syntax ] Parse : add support for TupleType and FunctionType
Merge remote - tracking branch ' origin / master ' into master - next
Merge pull request from Catfish - Man / bridgecleanup
[ Sema ] Disable long running nil coalescing perf test ( rdar : / / problem / 38963783 )
Fix protocol type mangling harder .
Introduce a simple dedicated SIL pass to fetch and use pre - specializations at - Onone .
Add the frontend option - disable - sil - perf - optzns .
Merge remote - tracking branch ' origin / master ' into master - next
[ swiftc ( 58 vs . 5396 ) ] Add crasher in swift : : OptionalType : : get ( . . . )
Merge pull request from shajrawi / licm_fix_test
Add Interpreter test using NSString bridging .
ASTDemangler : Add support for lowered function types
Merge pull request from swiftix / open - existential - fixes
[ test ] Update the API and ABI expected checks
Merge remote - tracking branch ' origin / master ' into master - next
Rename EnableLocalStoreDSE to DisableLocalStoreDSE and flip the condition
Scrap the custom allocator until we have time to fix it .
[ Serialization ] Move helper lambda into DeclDeserializer
Merge remote - tracking branch ' origin / master ' into master - next
[ incrParse ] Fix swift - syntax - test not complaining about unexpected reparsed functions
[ benchmark ] [ driver ] Teach the Benchmark_Driver how to parse . / Benchmark_O { , none } - - list now that tags are output as well .
Merge remote - tracking branch ' origin / master ' into master - next
stdlib : ranges : remove duplicate code
Merge pull request from huonw / emit - loaded - module - trace
Fix < rdar : / / problem / 24267414 > QoI : error : cannot convert value of type ' Int ' to specified type ' Int '
Add a runtime test for ObjC interop .
Merge remote - tracking branch ' origin / master ' into master - next
[ SourceKit ] Add test case for crash triggered in swift : : CompleteGenericTypeResolver : : resolveDependentMemberType ( swift : : Type , swift : : DeclContext * , swift : : SourceRange , swift : : ComponentIdentTypeRepr * )
Merge pull request from aschwaighofer / fix_opaque_indirect_return_results_loadable_by_address
Merge pull request from gottesmm / pr - 39539664cf3e8eeadf30363e1f27d58cca0974fb
Merge remote - tracking branch ' origin / master ' into master - next
[ migrator ] Handle simple renames for TypeDecl
Merge remote - tracking branch ' origin / master ' into master - next
Generalize the @ noescape attribute to be a type attribute allowed
[ Deserialization ] Verify that the module that PM handles is valid .
Release note NS_OPTIONS being default constructible .
Merge remote - tracking branch ' origin / master ' into master - next
introducing concrete initializers from all floating point types
Generic Specializer : Use getResilienceExpansion ( ) throughout ReabstractionInfo
[ build - script ] Tie llvm , swift , and lldb to the same sysroot
Merge pull request from compnerd / static - analysis
Add missing case when setting errors on a pattern .
[ driver ] Properly implemented MergeModule : : constructJob ( ) .
Merge remote - tracking branch ' origin / master ' into master - next
Fix Runtime . swift . gyb failur .
Merge pull request from jckarter / reflection - extra - inhabitants
Merge pull request from PatrickPijnappel / min - max
[ Type checker ] Make sure that the omit - needless - words logic validates decls .
Merge remote - tracking branch ' origin / master ' into master - next
Fix typo in CHECK prefixes .
[ codemotion ] Check for decrements between retain , cond_br not retain , enum_is_tag
Merge pull request from nkcsgexi / api - diff - enh - revert
[ python ] Use PEP - 0008 compliant code headers
Merge pull request from compnerd / everyone - gets - a - standard
Merge pull request from lanza / android
Handle optional return for UIGraphicsGetImageFromCurrentImageContext too .
Merge pull request from compnerd / order - order - order
Merge pull request from modocache / cmake - plist - info - apple - platforms
SILGen : Fix bug with local function default arguments
wire up initialization of argument variables with BBArguments that correspond
Merge remote - tracking branch ' origin / master ' into master - next
remove a mapping we don ' t need , at least so long as the swift AST doesn ' t become a DAG !
[ cmake ] Do not set CMAKE_INCLUDE_CURRENT_DIR .
Use correct formatting for start - of - file header .
Merge pull request from slavapestov / class - resilience - part - 1
Sema : Tweak VarDecl validation circularity break
Add a regression test for function argument conversions .
Removed repetitive docstring
Extend @ availability ( * , unavailable ) checking to protocol methods .
Remove a fatalError ( ) that I pasted here by mistake
AST : friend SimpleRequest without type specifier
[ SIL Diagnostics ] Create a mandatory pass to check correct usage of
Merge remote - tracking branch ' origin / master ' into master - next
[ Type checker ] Allow forward references to local types .
[ all - + 0 ] Add a new convention for setters : DefaultSetterConventions .
Merge remote - tracking branch ' origin / master ' into master - next
Make the parser reject ( with a fixit ) use of the @ weak / @ unowned attributes ,
Merge remote - tracking branch ' origin / master ' into master - next
Update AnyHashable documentation
Change remaining ' REQUIRES : OS = linux - gnu ' crashers to ' REQUIRES : deterministic - behavior '
[ swiftc ] Add test case for crash triggered in swift : : AbstractStorageDecl : : isGetterMutating ( ) const
[ build ] Use xcode - select to find Xcode instead of using / Applications / Xcode . app
Merge remote - tracking branch ' origin / master ' into master - next
[ NFC ] getWitnessMethodConformanceOrNone - > getWitnessMethodConformanceOrInvalid
Merge pull request from ddunbar / rdar26928189
[ stdlib ] JAFIA ( )
Merge pull request from marcrasi / textowner - sized - deallocation
Remove unused TypeVariableType : : getBaseBeingSubstituted ( ) . NFC
Never inline ` isConcrete_false ` test
Merge remote - tracking branch ' origin / master ' into master - next
Merge pull request from eeckstein / licm
Merge remote - tracking branch ' origin / master ' into master - next
Merge pull request from practicalswift / sourcekit - 061 - swift - constraints - solution - coercetotype
Restructure invalid parameter diagnostic
[ driver ] Add support for passing arguments after - - down to the frontend as - is if the driver is in immediate mode .
Merge remote - tracking branch ' origin / master ' into master - next
[ CodeCompletion ] Workaround a bunch of issues with ErrorType in completion
Merge pull request from moiseev / public - starttime
Merge pull request from jrose - apple / finding - ways - to - scope
Remove overfactoring and restore symmetry for - emit - objc - header - path
Fix an EscapeAnalysis assert to handle recent changes .
Merge remote - tracking branch ' origin / master ' into master - next
Merge pull request from danra / prefixmappatch1
Merge remote - tracking branch ' origin / master ' into master - next
Debug Info : Fix the lowering of the SILDebugScope tree to the LLVM
IRGen : Push SILType through a couple layers of partial application .
ClangImporter : refactor ClangImporterOptions to be ASTContext - owned . NFC
Merge remote - tracking branch ' origin / master ' into master - next
[ stdlib ] [ swift - 3 - indexing - model ] fixing stdlib / Generator tests
Merge pull request from nathawes / r54310026 - usr - generation - crash - in - invalid - code
Merge pull request from ematejska / merge - - to - 3 . 1 - branch
[ swift - ide - test ] Add option to allow providing - Xcc options .
Revert " [ XFail ] Disable Runtime / linux - fatal - backtrace ( 55490694 ) "
