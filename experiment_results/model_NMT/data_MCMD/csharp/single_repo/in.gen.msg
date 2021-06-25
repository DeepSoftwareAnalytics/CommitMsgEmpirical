Merge pull request from monojenkins / sync - pr - 34600 - from - runtime
2010 - 04 - 26 Jb Evain < jbevain @ novell . com >
[ profiler ] Clarify docs for mono_profiler_set_sample_mode ( ) .
Warnings cleanup
2007 - 01 - 06 Amit Biswas < amit @ amitbiswas . com >
* dis - cil . c ( get_encoded_user_string ) : use g_strescape instead of
2004 - 04 - 15 Atsushi Enomoto < atsushi @ ximian . com >
Removed classref signature
[ mdoc ] Fix typo
Add missing ForestTrustDomainInformation
2008 - 01 - 04 Rodrigo Kumpera < rkumpera @ novell . com >
svn path = / trunk / mcs / ; revision = 67913
2008 - 07 - 04 Zoltan Varga < vargaz @ gmail . com >
Revert " [ Fix ] Fix the computation of token sizes in ppdb files , the PDB stream has an additional table which contains the LGPL
2010 - 04 - 15 Rodrigo Kumpera < rkumpera @ novell . com >
2003 - 08 - 08 Nick Drochak < ndrochak @ gol . com >
undo r54467 changes
Merge pull request from BrzVlad / fix - major - log2
2001 - 07 - 17 Patrik Torstensson < patrik . torstensson @ labs2 . com >
[ xbuild ] Add missing CompareOrdinal method
Temporary checked - in repackaged admin app . TODO : fix repackaging issue , or add support for so - called " flat webapp " inside GH build system .
2008 - 10 - 27 Robert Jordan < robertj @ gmx . net >
2006 - 01 - 06 Gonzalo Paniagua Javier < gonzalo @ ximian . com >
Bump ikvm
2004 - 05 - 05 Anirban Bhattacharjee < banirban @ novell . com >
Warnings cleanup
2005 - 04 - 15 Cesar Lopez Nataren < cnataren @ novell . com >
* locales . c : nullify the ICU_collator member of CompareInfo when it is
2004 - 02 - 16 Atsushi Enomoto < atsushi @ ximian . com >
[ sgen ] Precisely log the size of the promoted objects during minors
Fix make package .
2009 - 03 - 04 Zoltan Varga < vargaz @ gmail . com >
New test .
2008 - 06 - 08 Zoltan Varga < vargaz @ gmail . com >
big changes to ObjectType from Mainsoft Java code . Cleanup in other files . several more complete
2008 - 12 - 10 Rodrigo Kumpera < rkumpera @ novell . com >
2006 - 10 - 07 Atsushi Enomoto < atsushi @ ximian . com >
makes sample cleaner as well
[ jit ] Fix the handling of byref locals in mono_allocate_stack_slots2 ( ) .
[ profiler ] Add a peephole environment ( )
* main . c ( dis_method_list ) : Oops . Check method_context for null .
Merge pull request from bassamtabbara / master
2005 - 09 - 16 Sebastien Pouliot < sebastien @ ximian . com >
A new type changes
2008 - 03 - 19 George Giolfan < georgegiolfan @ yahoo . com >
2008 - 06 - 10 Jb Evain < jbevain @ novell . com >
2009 - 04 - 18 Zoltan Varga < vargaz @ gmail . com >
add test
2005 - 01 - 11 Gonzalo Paniagua Javier < gonzalo @ ximian . com >
Small fixes for TARGET_JVM test scripts .
2004 - 11 - 02 Ben Maurer < bmaurer @ users . sourceforge . net >
* TabControl . cs : Remove debug code .
[ xbuild ] Implement GetDirectoryNameOfFileAbove ( )
2004 - 07 - 15 Zoltan Varga < vargaz @ freemail . hu >
2004 - 06 - 11 Sebastien Pouliot < sebastien @ ximian . com >
[ runtime ] Use mono_debug_method_lookup_location ( ) instead of mono_debug_symfile_lookup_location ( ) , the latter doesn', "'", 't support portable pdb files . Use mono_debug_free_source_location ( ) for freeing . Fixes . ( )
2005 - 11 - 20 Chris Toshok < toshok @ ximian . com >
2004 - 07 - 30 Sebastien Pouliot < sebastien @ ximian . com >
Additional test cases for icons . Test size 48x48 , 64x64 and 96x96 . Test monochrome and 256 colors icons .
2006 - 07 - 14 Zoltan Varga < vargaz @ gmail . com >
2008 - 02 - 11 Marek Safar < marek . safar @ gmail . com >
[ System . Core ] Disable tests for netstandard20 APIs
* main . c : Spaces after visibility types .
Merge pull request from BrzVlad / fix - complex - conc - scan
[ wasm ] Adjust . c ( )
2007 - 05 - 31 Marek Habersack < mhabersack @ novell . com >
2005 - 05 - 11 Lluis Sanchez Gual < lluis @ novell . com >
* HttpRequest . cs : Cleanup method .
Update
* DataGridColumnStyle . cs ( SetDataGrid ) : don', "'", 't try to access a null
2005 - 03 - 13 Robert Jordan < robertj @ gmx . net >
oops
Tue Jul 28 11 : 34 : 49 CEST 2002 Paolo Molaro < lupus @ ximian . com >
* SoapAttribute . cs , SoapFieldAttribute . cs , SoapMethodAttribute . cs ,
[ sgen ] Precisely log the size of the promoted objects during minors
2008 - 06 - 26 Marek Safar < marek . safar @ gmail . com >
2009 - 07 - 04 Atsushi Enomoto < atsushi @ ximian . com >
[ runtime ] Work around an ios9 problem where dlopen ( ) cant', "'", 'find system libraries specified without paths .
Merge pull request from lambdageek / dev / monoerror - stringy
[ runtime ] Change the code in process_call ( ) a bit .
interp / TlsSetValue builds
2004 - 02 - 01 Zoltan Varga < vargaz @ freemail . hu >
[ xbuild ] Fix Tooltask . ToolPath ( )
2006 - 05 - 04 Sureshkumar T < tsenganal @ novell . com >
Thu May 16 16 : 08 : 54 CEST 2002 Paolo Molaro < lupus @ ximian . com >
More test .
2004 - 10 - 13 Sebastien Pouliot < sebastien @ ximian . com >
[ jit ] Fix the handling of byref locals in mono_allocate_stack_slots2 ( ) .
2006 - 04 - 28 Chris Toshok < toshok @ ximian . com >
[ mcs ] Update failing test
[ threadpool - ms ] Fix incorrect use of mono_rand .
* MethodInstr . cs : callvirt operation is implicitly instance .
Collection < T > : Validate parameters in the public methods ( )
2006 - 10 - 02 Dick Porter < dick @ ximian . com >
[ corlib ] First specific culture returned by enumerator was not fully initialized . Fixes
2003 - 12 - 21 Ravi Pratap < ravi @ ximian . com >
2003 - 11 - 13 Zoltan Varga < vargaz @ freemail . hu >
2004 - 12 - 17 Martin Baulig < martin @ ximian . com >
Add new error
[ System . ServiceModel ] Changed test port from 8080 to 30158
[ jit ] Fix an unreachable code part in mini_emit_memset ( ) . Fixes . Thanks to Bill Seurer for spotting it .
* docs . cs ( MyMenuAdapter ) : Add new forwarding methods
[ wasm ] Disable the AOT cache check on wasm as well . ( )
[ wasm ] [ debugger ] Clear valuetypes cache on Debugger . resume ( )
[ llvmonly ] Remove some dead code .
[ msvc ] Update csproj files
[ csproj ] Update project files
2009 - 08 - 30 Zoltan Varga < vargaz @ gmail . com >
[ aprofutil ] Fix THISDIR location in a few projects , so genproj can generate MSBuild files
* Mono . Documentation / monodocer . cs : Only provide a Main ( ) method and
use 32bpp bitmaps internally
Fix a testcase
2002 - 03 - 15 Duncan Mak < duncan @ duncan @ ximian . com >
System . Drawing . vmwcsproj : Added SystemIcons . jvm . cs , excluded SystemIcons . cs
[ mobile_static ] Update BCL Linked Size
2005 - 01 - 15 Miguel de Icaza < miguel @ novell . com >
[ arm ] Fix crashes on ios9 by using 16kb trampoline pages .
Merge pull request from akoeplinger / pupitetris - Bug - 35635
add a couple more tests
TestRegion . cs : added test cases for matrix operations on infinite regions ( result is always an infinite region ) . Add test cases for scale and scale + translate on regions ( to see that they do not trigger a convertion into bitmap - based regions ) .
Merge pull request from esdrubal / xmlgenerator
Merge pull request from esdrubal / servicemodel
2007 - 02 - 14 Marek Sieradzki < marek . sieradzki @ gmail . com >
First work on System . Workflow . Runtime
Added new files
2004 - 05 - 22 Gonzalo Paniagua Javier < gonzalo @ ximian . com >
Update update
Fix compilation errors discovered with csc . exe of FW 2 . 0 SP1
Merge pull request from Unity - Technologies / fix - isuserprotected
[ sgen ] Track down promoted size also with parallel minors
Tue Mar 16 15 : 47 : 16 CET 2006 Paolo Molaro < lupus @ ximian . com >
[ runtime ] Added - - with - cross - offsets configure option for including an explicit AOT cross offsets file .
2009 - 04 - 16 Zoltan Varga < vargaz @ gmail . com >
Document layout
2009 - 11 - 18 Mark Probst < mark . probst @ gmail . com >
2002 - 06 - 14 Duncan Mak < duncan @ ximian . com >
* statement . cs ( Block . CloneTo ) : Clone Toplevel field too .
Add retrying of failed tests by default to reduce flaky tests ( )
Add these files
[ reflection ] Harder the custom attribute code against common type load errors .
* ThemeWin32Classic . cs : When imagelists are used the image area is
Flush
2006 - 02 - 27 Atsushi Enomoto < atsushi @ ximian . com >
2010 - 01 - 05 Miguel de Icaza < miguel @ novell . com >
* DateTimePicker . cs : Fixed a NRE if CustomFormat was null . Handles the
* * * empty log message * * *
Final changes
[ runtime ] Use mono_debug_method_lookup_location ( ) instead of mono_debug_symfile_lookup_location ( ) , the latter doesn', "'", 't support portable pdb files . Use mono_debug_free_source_location ( ) for freeing . Fixes . ( )
2007 - 07 - 17 Sebastien Pouliot < sebastien @ ximian . com >
2003 - 07 - 12 Dick Porter < dick @ ximian . com >
2006 - 11 - 16 Zoltan Varga < vargaz @ gmail . com >
Simplified DataSetTests from Mainsoft
2004 - 11 - 04 John Luke < john . luke @ gmail . com >
from head
2002 - 05 - 12 Ville Palo < vi64pa @ koti . soon . fi >
Merge pull request from esdrubal / fileinfo_moveto
[ runtime ] Use mono_set_pending_exception ( ) in some places in icalls .
* BuildTask . cs ( Execute ) : Use task_logger only after the task
Add new files
fix test .
2005 - 12 - 05 Ben Maurer < bmaurer @ ximian . com >
[ reflection ] Add support for redirected StandardInput writes . Fixes
2005 - 07 - 17 Atsushi Enomoto < atsushi @ ximian . com >
[ arm ] Fix crashes on ios9 by using 16kb trampoline pages .
new test
[ runtime ] Fix build when using musl libc .
Tue Aug 1 16 : 47 : 15 CEST 2001 Paolo Molaro < lupus @ ximian . com >
2006 - 05 - 09 Sebastien Pouliot < sebastien @ ximian . com >
2009 - 03 - 04 Rodrigo Kumpera < rkumpera @ novell . com >
2005 - 10 - 03 Ben Maurer < bmaurer @ ximian . com >
[ corlib ] Default to fast Span < T > ( )
2009 - 04 - 28 Jb Evain < jbevain @ novell . com >
2006 - 09 - 01 Senganal T < tsenganal @ novell . com >
Merge pull request from kumpera / unlimited - abort
Add missing dist entries .
2005 - 01 - 10 Peter Bartok < pbartok @ novell . com >
Fix a regression introduced by db4299a1 .
[ llvmonly ] Remove some dead code .
2008 - 11 - 20 Martin Baulig < martin @ ximian . com >
2004 - 07 - 09 Zoltan Varga < vargaz @ freemail . hu >
2002 - 09 - 03 Duncan Mak < duncan @ ximian . com >
2002 - 03 - 17 Jeffrey Stedfast < fejj @ ximian . com >
New tests .
New test .
2009 - 08 - 18 Rodrigo Kumpera < rkumpera @ novell . com >
Fix ChangeLog entry .
fix credits
2002 - 12 - 09 Miguel de Icaza < miguel @ ximian . com >
Update
Dear God , make me recompile my code after removing debug * before * I check in
2008 - 12 - 08 Zoltan Varga < vargaz @ gmail . com >
2002 - 03 - 21 Dietmar Maurer < dietmar @ ximian . com >
ChangeLog : Updated .
add new links
[ ci ] Add Mono . Cecil . * . Has * checks when comparing methods / types
[ netcore ] Re - enable some BinaryFormatter tests ( )
2007 - 01 - 09 Marek Safar < marek . safar @ gmail . com >
Simplified DataSetTests from Mainsoft
* Thread . cs : Implement InnerText setter .
Update sgen - PARTY - NOTICES . TXT
2006 - 11 - 25 Atsushi Enomoto < atsushi @ ximian . com >
[ Mono . Debugger . Soft ] Catching AssemblyMirror in AppDomainMirror and TypeMirror in AssemblyMirror
this changes
2004 - 03 - 08 Atsushi Enomoto < atsushi @ ximian . com >
2003 - 09 - 01 Miguel de Icaza < miguel @ ximian . com >
oops
[ runtime ] Fix a warning .
2007 - 07 - 11 Marek Habersack < mhabersack @ novell . com >
2005 - 06 - 13 Sureshkumar T < tsureshkumar @ novell . com >
[ jit ] Emit the seq point code used for soft breakpoints .
2009 - 08 - 04 Zoltan Varga < vargaz @ gmail . com >
Bump ikvm
2005 - 09 - 20 Zoltan Varga < vargaz @ gmail . com >
2008 - 09 - 02 Marek Safar < marek . safar @ gmail . com >
2006 - 09 - 30 Atsushi Enomoto < atsushi @ ximian . com >
[ Fixes ]
2004 - 11 - 06 Martin Baulig < martin @ ximian . com >
2009 - 07 - 09 Zoltan Varga < vargaz @ gmail . com >
2009 - 09 - 01 Zoltan Varga < vargaz @ gmail . com >
refactoring
2008 - 03 - 03 Jonathan Pobst < monkey @ jpobst . com >
Merge pull request from akoeplinger / pupitetris - Bug - 35635
new test
Revert r84984 and r85314
2002 - 03 - 28 Rodrigo Moya < rodrigo @ ximian . com >
2005 - 01 - 09 Sebastien Pouliot < sebastien @ ximian . com >
Populate properties of AssemblyLoadStart / Stop events ( dotnet / coreclr )
Fix the build .
removed test file
2007 - 11 - 08 Zoltan Varga < vargaz @ gmail . com >
[ mono - error ] Introduce MonoError into mono_metadata_parse_generic_param .
2007 - 10 - 30 Carlos Alberto Cortez < calberto . cortez @ gmail . com >
more docu
Fix
2006 - 04 - 29 Igor Zelmanovich < igorz @ mainsoft . com >
[ wasm ] Remove unused icalls in process_breakpoint .
2002 - 09 - 22 Nick Drochak < ndrochak @ gol . com >
[ profiler ] Add dedicated coverage profiler ( )
2008 - 08 - 06 Marek Safar < marek . safar @ gmail . com >
svn path = / trunk / mcs / ; revision = 51213
2009 - 05 - 02 Gonzalo Paniagua Javier < gonzalo @ novell . com >
[ jit ] Precompute the seq point code used by soft breakpoints .
[ runtime ] Make sure MONO_CONTEXT_GET_CURRENT ( ) sets sp to an aligned value on watchos .
[ runtime ] Use mono_set_pending_exception ( ) in some places in icalls .
[ sgen ] Fix pinning in the presence of canaries .
2006 - 05 - 06 Gonzalo Paniagua Javier < gonzalo @ ximian . com >
[ threads ] Fix Posix async resume ( )
2004 - 12 - 15 Martin Baulig < martin @ ximian . com >
Backport
2009 - 04 - 14 Carlos Alberto Cortez < calberto . cortez @ gmail . com >
2004 - 06 - 15 Marek Safar < marek . safar @ seznam . cz >
2009 - 08 - 06 Atsushi Enomoto < atsushi @ ximian . com >
[ dlr ] Clenup warnings
[ sgen ] Fix a regression caused by 07fc727b8698a100aeb214ae3f7eb1cc3f5bab20 . Fixes the windows build .
2004 - 04 - 30 Gonzalo Paniagua Javier < gonzalo @ ximian . com >
2004 - 09 - 03 Martin Baulig < martin @ ximian . com >
* Mono . Mozilla / DOM / Navigation . cs : Fix calls to the nsIWebNavigation
In Test / Microsoft . Build . Tasks :
2009 - 04 - 23 Gonzalo Paniagua Javier < gonzalo @ novell . com >
* HttpWebClientProtocol . cs : Obey rules for fixed sized windows ( no
2005 - 12 - 15 Lluis Sanchez Gual < lluis @ novell . com >
2010 - 02 - 19 Carlos Alberto Cortez < calberto . cortez @ gmail . com >
[ ci ] Enable linker tests on PRs
Bump roslyn ( to include more files )
Merge pull request from esdrubal / fileinfo_moveto
* locales . c ( dis_directive_assemblyref ) : Dump custom attributes .
Initialization of default proxy added .
2010 - 05 - 23 Rodrigo Kumpera < rkumpera @ novell . com >
[ ci ] Enable linker tests on PRs
Update
Use _vscprintf ( ) for mingw - w64 compilers
2009 - 01 - 20 Marek Habersack < mhabersack @ novell . com >
2009 - 05 - 25 Sebastien Pouliot < sebastien @ ximian . com >
2004 - 05 - 23 Marek Safar < marek . safar @ seznam . cz >
2007 - 07 - 04 William Holmes < billholmes54 @ gmail . com >
[ mobile_static ] AOT libs not aoted by the test runner
2002 - 08 - 04 Gonzalo Paniagua Javier < gonzalo @ ximian . com >
[ System . EnterpriseServices ] Rename method parameters to match . NET contract
Fix the previous marshalling change so it only applies to chars , not structures which contain char fields .
* * * empty log message * * *
Ignore right mouse clicks in toolbar
2006 - 01 - 22 Sebastien Pouliot < sebastien @ ximian . com >
[ profiler ] Enable the root_type size code to uleb128 .
[ xBuild ] Add ItemDefinitionGroup basic handling
2005 - 12 - 19 Atsushi Enomoto < atsushi @ ximian . com >
* Directory . cs : Add named param table and methods to get named param
2006 - 08 - 04 Chris Toshok < toshok @ ximian . com >
Fix inacurrate reference
[ jit ] Fix . c . ( )
bootstrap fix
[ runtime ] Remove pre WinXP SP2 windows support code .
Checking array elements is done using ImplicitReferenceConversionExists
[ System . Net . Http ] Incomplete byte ranges parsing . Fixes
2009 - 04 - 15 Rodrigo Kumpera < rkumpera @ novell . com >
[ System . Data . SqlClient ] Avoid " invalid source table " exception in bulk inserts
2005 - 06 - 27 Zoltan Varga < vargaz @ gmail . com >
Initialization of default proxy added .
win32 resource name is changed
2004 - 04 - 29 Marek Safar < marek . safar @ seznam . cz >
* XmlReflectionImporterTests . cs : Fixed rounding for float and the missing output now includes all the precision ( not counting preceding zeros ) . Add test cases for scale and scale + translate on regions ( to see that they do not trigger a convertion into bitmap - based regions ) .
Add new files
[ tests ] Xamarin segfaults on Android , probably due to AppDomain . Unload
2006 - 01 - 04 Gonzalo Paniagua Javier < gonzalo @ ximian . com >
- Part of the previus checkin , missed these files
removed test file
2003 - 10 - 17 Ben Maurer < bmaurer @ users . sourceforge . net >
[ asp . net ] Do nothing if null SessionStateStoreData is passed to SessionStateServerHandler . SetAndReleaseItemExclusive
[ runtime ] Move Mono . Cecil . SymbolWriter to EXTRA_DIST
[ mini ] Run MONO_MMAP_32BIT . ( )
2008 - 01 - 16 Zoltan Varga < vargaz @ gmail . com >
[ Coop ] ves_icall_RuntimeFieldInfo_GetRawConstantValue . ( )
New test .
[ csproj ] Update csproj files
2005 - 12 - 01 Martin Baulig < martin @ ximian . com >
fix bug
2005 - 06 - 07 Gonzalo Paniagua Javier < gonzalo @ ximian . com >
2008 - 03 - 17 Marek Safar < marek . safar @ gmail . com >
Rename parameter back .
minor readability improvements
Merge pull request from ninjarobot / fix - xml - ser - tests
2008 - 11 - 16 Jonathan Pobst < monkey @ jpobst . com >
Merge pull request from lateralusX / lateralusX / add - waitforkeypress - test - project
- Fixed Text get property ( also fixes bug )
* main . c ( dis_method_list ) : Oops . Check method_context for null .
2006 - 06 - 02 Alexander Olk < alex . olk @ googlemail . com >
basic summaries
2007 - 03 - 23 Nidhi Rawal < rawalnidhi_rawal @ yahoo . com >
2006 - 03 - 13 Gonzalo Paniagua Javier < gonzalo @ ximian . com >
* ILParser . jay : Same as below but with float64 ( )
2003 - 01 - 11 Nick Drochak < ndrochak @ gol . com >
2006 - 11 - 16 Carlos Alberto Cortez < calberto . cortez @ gmail . com >
[ llvmonly ] Transform the code returned by init_method ( ) into an exception and throw it , it can be an exception thrown by a cctor .
from head
[ jit ] Get rid of the create_delegate_trampoline_data ( ) helper function which only has one caller .
* XmlReflectionImporterTests . cs : Added ImportTypeMapping tests for
In tests :
[ Coop ] ves_icall_RuntimeFieldInfo_GetRawConstantValue . ( )
2004 - 09 - 08 Umadevi S < sumadevi @ novell . com >
2010 - 03 - 16 Atsushi Enomoto < atsushi @ ximian . com >
2005 - 02 - 09 Lluis Sanchez Gual < lluis @ novell . com >
2008 - 10 - 04 Mark Probst < mark . probst @ gmail . com >
New test .
2005 - 11 - 30 Sebastien Pouliot < sebastien @ ximian . com >
[ Socket ] Move remaining icalls and struct from Socket_2_1 . cs to Socket . cs
2010 - 01 - 19 Rodrigo Kumpera < rkumpera @ novell . com >
Avoid copying the stack alignment area during tail calls on x86 , if the caller method is frm LLVM , it might not have that area .
2003 - 11 - 22 Gonzalo Paniagua Javier < gonzalo @ ximian . com >
New tests .
2009 - 09 - 14 Zoltan Varga < vargaz @ gmail . com >
2003 - 07 - 04 Gonzalo Paniagua Javier < gonzalo @ ximian . com >
* EventLog . cs : Minor fixes .
Little fix
Fixed SGen qsort code to compile with MSVC .
[ sgen ] Make unnecessary stack check into assertion .
2008 - 03 - 13 Zoltan Varga < vargaz @ gmail . com >
Updated .
Add DateTimeOffsetConverter . cs to DateTimeOffsetConverterTests . cs .
use Invariant culture for string manipulations
[ llvm ] Add a _checked version for mono_class_from_name ( ) .
[ interp ] Add intrinsics for Math operations ( )
2008 - 03 - 22 Jb Evain < jbevain @ novell . com >
[ threadpool - ms ] Fix incorrect use of mono_rand .
[ wasm ] Small fixes . ( )
2001 - 12 - 21 Miguel de Icaza < miguel @ ximian . com >
[ mini ] Fix CAS atomic ordering under LLVM .
2009 - 10 - 29 Miguel de Icaza < miguel @ novell . com >
2008 - 07 - 28 Marek Safar < marek . safar @ gmail . com >
[ acceptance - tests ] Bump coreclr to pick up 3b2969dffe191b03f1ac85a9c1e62417c9b7833b :
2006 - 04 - 09 Gonzalo Paniagua Javier < gonzalo @ ximian . com >
- Duh !
[ corlib ] Fix TimeZoneInfo tests ( )
2007 - 10 - 05 Alp Toker < alp @ atoker . com >
2004 - 06 - 09 Gonzalo Paniagua Javier < gonzalo @ ximian . com >
2006 - 05 - 16 Alexander Olk < alex . olk @ googlemail . com >
Add new tests
[ coop ] Fixed GCResizeEvent value overflow ( )
2006 - 02 - 17 Atsushi Enomoto < atsushi @ ximian . com >
[ build ] Try to enable parallel make on RPM build
First PrintDialog test .
2005 - 02 - 02 Sebastien Pouliot < sebastien @ ximian . com >
[ llvm ] Emit the code in process_call ( ) a bit .
2006 - 07 - 03 Marek Safar < marek . safar @ seznam . cz >
[ Coop ] ves_icall_RuntimeFieldInfo_GetRawConstantValue . ( )
* cs1502 . cs ( ComposedCast ) : Check for arrays of TypedReference
moved to its own module
2002 - 07 - 17 Martin Baulig < martin @ gnome . org >
2008 - 07 - 10 Zoltan Varga < vargaz @ gmail . com >
Bump roslyn ( to include more files )
2002 - 11 - 02 Miguel de Icaza < miguel @ ximian . com >
* Mono . Posix . dll . sources : Remove Mono . Unix obsolete files .
2008 - 10 - 21 Jonathan Pobst < monkey @ jpobst . com >
2003 - 11 - 18 Gonzalo Paniagua Javier < gonzalo @ ximian . com >
Minor improvements to the new moonlight naming convention .
Tweak the Unreal profile
[ xbuild ] Fix bug .
[ corlib ] Interning of utc literals in TimeZoneInfo conversion . Fixes
[ runtime ] Make WaitForPendingFinalizers ( ) really wait for the finalizers added by a previous GC . Fixes .
Implemented NullTarget ( )
2002 - 12 - 17 Rachel Hestilow < hestilow @ ximian . com >
2003 - 05 - 10 Ville Palo < vi64pa @ kolumbus . fi >
2005 - 05 - 04 Gonzalo Paniagua Javier < gonzalo @ ximian . com >
2006 - 07 - 18 Sebastien Pouliot < sebastien @ ximian . com >
[ System ] Remove duplicate files
Flush
. gitignore
2004 - 09 - 09 Atsushi Enomoto < atsushi @ ximian . com >
[ sgen ] Wrap DynArray into strongly types variants making the code cleaner and safer .
Final changes
2004 - 03 - 12 Atsushi Enomoto < atsushi @ ximian . com >
Flush
* EventLog . cs : Fixed Designer attribute to match MS . NET .
2003 - 04 - 10 Gonzalo Paniagua Javier < gonzalo @ ximian . com >
2008 - 11 - 17 Rodrigo Kumpera < rkumpera @ novell . com >
Merge pull request from lambdageek / fix - gc - descr - object - c
Merge pull request from akoeplinger / fix - ipv6 - teredo
2005 - 07 - 10 Miguel de Icaza < miguel @ novell . com >
2004 - 03 - 03 Zoltan Varga < vargaz @ freemail . hu >
2002 - 05 - 12 Gonzalo Paniagua Javier < gonzalo @ ximian . com >
svn path = / trunk / mcs / ; revision = 67913
* ServiceContainerTest . cs : Added test for .
2006 - 02 - 01 Marek Sieradzki < marek . sieradzki @ gmail . com >
Bump ikvm
2005 - 10 - 24 Martin Baulig < martin @ ximian . com >
Remove Console . WriteLine .
2004 - 09 - 03 Marek Safar < marek . safar @ seznam . cz >
2005 - 10 - 29 Gonzalo Paniagua Javier < gonzalo @ ximian . com >
[ interp ] Add intrinsics for Math operations ( )
[ jit ] Fix a regression introduced by : ( )
2009 - 06 - 25 Alp Toker < alp @ atoker . com >
2008 - 03 - 25 Marek Safar < marek . safar @ gmail . com >
2003 - 10 - 14 Gonzalo Paniagua Javier < gonzalo @ ximian . com >
Fix a warning .
[ wasm ] Add bidi binding for Promise / Task .
2007 - 03 - 11 Atsushi Enomoto < atsushi @ ximian . com >
2006 - 02 - 17 Marek Safar < marek . safar @ seznam . cz >
2005 - 01 - 12 Geoff Norton < gnorton @ customerdna . com >
2002 - 12 - 02 Gonzalo Paniagua Javier < gonzalo @ ximian . com >
[ jit ] Add myself as author .
ChangeLog : Updated ChangeLog .
[ docs ] Fix UnicodeEncoding GetMaxByteCount and GetMaxCharCount to return the same values as . net .
Prepare to reveal defect in XmlSerializer ( Bug 18558 )
Add ExceptionDispatchInfo . SetCurrentStackTrace ( dotnet / coreclr )
* * * * Merged r37266 from MCS * * * *
[ runtime ] Fix the remoting build .
Optimize box ( X ) = = null to false
2008 - 08 - 01 Jb Evain < jbevain @ novell . com >
2010 - 04 - 14 Atsushi Enomoto < atsushi @ ximian . com >
[ System . Private . CoreLib ] Add Nullable runtime support
syncing
svn path = / trunk / mcs / ; revision = 53502
[ btls ] : Make the', "'", 'MonoBtlsX509LookupMonoCollection', "'", 'logic consistent with the native implementation . ( )
* CrossAppDomainChannel . cs : In CreateMessageSink , ignore the url
Add TLS offset probing to mach - arm in the same way it', "'", 's done on x86 .
[ runtime ] Remove use of mono_raise_exception in mono_object_isinst_mbyref_checked
Implement libffi
[ System . Data . SqlClient ] Avoid " invalid source table " exception in bulk inserts
[ msvc ] Convert Console . IsPinnable . ( )
Add new unit tests for System . Security', "'", 's HostSecurityManager and PermissionSetCollection classes
- Add some [ Flags ] .
2004 - 03 - 17 Marek Safar < marek . safar @ seznam . cz >
Merge pull request from NoesisLabs / master
Handle long paths in jay
Merge pull request from Unity - Technologies / master - gc - race
2004 - 06 - 18 Francisco Figueiredo Jr . < fxjrlists @ yahoo . com . br >
undo .
Forgot to mark up new internal function prototype .
Bump roslyn ( )
[ Coop ] ves_icall_RuntimeFieldInfo_GetRawConstantValue . ( )
2006 - 02 - 01 Atsushi Enomoto < atsushi @ ximian . com >
Merge pull request from cmp - / fix - sdb - trampoline - windows - x64
* SerializationCodeGenerator . cs :
New test .
* CheckBoxList . cs :
[ llvm ] Add a new ICALL_ADDR_CALL patch type which is only used on calls , since ICALL_ADDR might be used without a call .
Check file
2007 - 09 - 20 Everaldo Canuto < everaldo @ simios . org >
Mon Jul 17 17 : 06 : 15 CEST 2007 Paolo Molaro < lupus @ ximian . com >
2004 - 08 - 30 Zoltan Varga < vargaz @ freemail . hu >
[ jit ] Fix the EqualityComparer . get_Default ( ) optimization for now , it seems to cause failures in the corlib tests on the ios simulator . ( )
2008 - 09 - 30 Marek Habersack < mhabersack @ novell . com >
Updated
2002 - 09 - 27 Gonzalo Paniagua Javier < gonzalo @ ximian . com >
Move AssemblyLoadContext . GetResolvedUnmanagedDll to shared ( )
* DateTimeTest . cs : Added test for bug .
2008 - 02 - 01 Ivan N . Zlatev < contact @ i - nz . net >
Fix inacurrate reference
2007 - 08 - 28 Zoltan Varga < vargaz @ gmail . com >
build fix
[ System ] : Cleanup internal TLS APIs in preparation of adding renegotiation support . ( )
[ profiler ] Fix the support for MONO_ARCH_NOTIFY_PENDING_EXC , it only works on amd64 , and its not worth the complexity .
2008 - 01 - 28 Sebastien Pouliot < sebastien @ ximian . com >
2007 - 11 - 06 Zoltan Varga < vargaz @ gmail . com >
[ runtime ] Register the locking in set_breakpoint ( ) a bit .
2002 - 12 - 04 Gaurav Vaish < gvaish @ iitk . ac . in >
2008 - 07 - 17 Rodrigo Kumpera < rkumpera @ novell . com >
2008 - 04 - 03 George Giolfan < georgegiolfan @ yahoo . com >
Get rid of the MONO_ARCH_ENABLE_LMF_IR define added by 0938c69a52654e02bbb8f0c2107183404516dec6 , use flags in MonoCompile instead .
2005 - 02 - 20 Lluis Sanchez Gual < lluis @ novell . com >
2008 - 05 - 25 Marek Safar < marek . safar @ gmail . com >
* TreeView . cs : Obey rules for fixed sized windows ( no
2008 - 04 - 13 Carlos Alberto Cortez < calberto . cortez @ gmail . com >
Fix
2008 - 05 - 10 Jonathan Pobst < monkey @ jpobst . com >
2010 - 01 - 09 Atsushi Enomoto < atsushi @ ximian . com >
2006 - 05 - 04 John Luke < john . luke @ gmail . com >
2009 - 07 - 24 Marek Habersack < mhabersack @ novell . com >
[ WindowsBase ] Make tests less verbose
Check test
[ tests ] Add test for apploading related crashes due to repetitive load / unload of the same assembly .
[ Tailcall ] Proper fix for F # regression . ( )
* * * empty log message * * *
[ Coop ] ves_icall_RuntimeFieldInfo_GetRawConstantValue . ( )
2005 - 11 - 04 Sebastien Pouliot < sebastien @ ximian . com >
2005 - 04 - 16 Jordi Mas i Hernandez < jordi @ ximian . com >
New tests .
2006 - 01 - 30 Martin Baulig < martin @ ximian . com >
Revert
add CallWithExpression tests
Small fix .
[ runtime ] Remove unused LLVM_MONO_BRANCH define .
Use _vscprintf ( ) for mingw - w64 compilers
2008 - 07 - 11 Miguel de Icaza < miguel @ novell . com >
2009 - 10 - 29 Jonathan Pobst < monkey @ jpobst . com >
SqlBulkCopy Implementation
2008 - 10 - 04 Sebastien Pouliot < sebastien @ ximian . com >
[ llvm ] Remove unused LLVM_MONO_BRANCH define .
2004 - 03 - 15 Juraj Skripsky < juraj @ hotfeet . ch >
* OutputCacheModule . cs : VaryBy keys now need the HttpContext not just the request .
2002 - 06 - 20 Sebastien Pouliot < spouliot @ videotron . ca >
Revert r46073 . It caused corlib_test . dll compilation to fail .
2008 - 01 - 24 Sebastien Pouliot < sebastien @ ximian . com >
* ILParser . jay : Create external method references
[ mcs ] Re - add obsolete reporting for enumerators removed in 6643bc119cbe1e94b5634e6452dd6eb40c7681c7
[ interp ] Rename variable
Implemented Encoder . cs .
[ sgen ] Fix invalid localized initialization marshaling ( )
[ jit ] Fix the support for MONO_ARCH_NOTIFY_PENDING_EXC , it only works on amd64 , and its not worth the complexity .
Style nits
2004 - 07 - 07 Miguel de Icaza < miguel @ ximian . com >
Merge pull request from ntherning / fix - assert - in - poll_event_wait - on - WSAENOTSOCK - on - windows
2006 - 03 - 13 Chris Toshok < toshok @ ximian . com >
[ xbuild ] Evaluate Import only if its condition evaluates to true .
[ sgen ] Get rid of the create_delegate_trampoline_data ( ) helper function which only has one caller .
Fix compile
* SolutionParser . cs : Add RealUser , RealGroup , EffectiveUser , EffectiveGroup
2003 - 03 - 10 Gonzalo Paniagua Javier < gonzalo @ ximian . com >
eh , open the floodgates . stop suppressing these warnings .
[ corlib ] First specific culture returned by enumerator was not fully initialized . Fixes
2005 - 06 - 22 Zoltan Varga < vargaz @ freemail . hu >
2006 - 07 - 08 Sebastien Pouliot < sebastien @ ximian . com >
2009 - 08 - 04 Rodrigo Kumpera < rkumpera @ novell . com >
[ ci ] Don', "'", 't build mobile profiles in a monolite build
2009 - 02 - 04 Atsushi Enomoto < atsushi @ ximian . com >
Fix warning .
* metadata . c ( mono_metadata_inflate_generic_inst ) : Remove .
2003 - 01 - 28 Tim Coleman < tim @ timcoleman . com >
2005 - 01 - 13 Martin Baulig < martin @ ximian . com >
New test .
Fix bug .
[ utils ] Add foreach variant that supports filtering the list as you go .
merge - r 58778 : 58779
Fixed', ['Update
2003 - 08 - 30 Gonzalo Paniagua Javier < gonzalo @ ximian . com >
2003 - 10 - 28 : Jean - Marc Andre < jean - marc . andre @ polymtl . ca >
2004 - 03 - 11 Atsushi Enomoto < atsushi @ ximian . com >
[ Cleanup ] Add foreach variant that supports filtering the list as you go .
. gitignore
add redundant
* * * empty log message * * *
Stupid tabs
[ Tailcall ] Fill in some missing pieces : x86reg , armreg , armmembase , align x86patch , x86aot , armhardfloat . ( )
2010 - 04 - 03 Atsushi Enomoto < atsushi @ ximian . com >
Tests
New test .
Move Timer to shared CoreLib partition . ( dotnet / coreclr )
[ interp ] Fix more MonoError reuse . ( )
TARGET_JVM : is fixed
add CallWithExpression tests
Building answer update .
* CompareInfo . cs : Marked UseManagedCollation internal to have API
some people claim they have old SYSMONO that cannot compile this source and demand to " fix " this code .
Test case for bug 42135 .
spec fixes
[ wasm ] [ debugger ] Clear valuetypes cache on Debugger . resume ( )
Add missing files
[ runtime ] Make WaitForPendingFinalizers ( ) really wait for the finalizers added by a previous GC . Fixes .
[ runtime ] Fix the build .
2002 - 07 - 28 Daniel Morgan < danmorg @ sc . rr . com >
fix HintPath
[ jit ] Remove some dead code / includes .
2002 - 04 - 17 Rafael Teixeira < rafaelteixeirabr @ hotmail . com >
2009 - 01 - 11 Zoltan Varga < vargaz @ gmail . com >
2007 - 03 - 03 Jonathan Pobst < monkey @ jpobst . com >
Merge pull request from Unity - Technologies / master - gc - race
New tests .
2007 - 04 - 22 Zoltan Varga < vargaz @ gmail . com >
2005 - 01 - 09 Marek Safar < marek . safar @ seznam . cz >
2005 - 03 - 14 Zoltan Varga < vargaz @ freemail . hu >
[ wasm ] [ debugger ] Hash messages and sessions directly ( )
2007 - 08 - 24 Everaldo Canuto < ecanuto @ novell . com >
[ mini ] Remove some dead code .
[ runtime ] mono_optimize_branches no longer breaks with OP_IL_SEQ_POINT .
2009 - 03 - 04 Marek Safar < marek . safar @ gmail . com >
[ btls ] : Make the', "'", 'MonoBtlsX509LookupMonoCollection', "'", 'logic consistent with the native implementation . ( )
[ sgen ] Fix a regression caused by 07fc727b8698a100aeb214ae3f7eb1cc3f5bab20 . Fixes the windows build .
eol
2005 - 01 - 07 Maverson Eduardo Schulze Rosa < maverson @ gmail . com >
2006 - 02 - 22 Atsushi Enomoto < atsushi @ ximian . com >
2006 - 11 - 03 Atsushi Enomoto < atsushi @ ximian . com >
[ profiler ] Fix the EqualityComparer . get_Default ( ) optimization for now , it seems to cause failures in the corlib tests on the ios simulator . ( )
2002 - 06 - 24 Martin Baulig < martin @ gnome . org >
Fix test suite
2004 - 06 - 15 Anirban Bhattacharjee < banirban @ novell . com >
2001 - 12 - 28 Miguel de Icaza < miguel @ ximian . com >
2008 - 05 - 06 Jonathan Pobst < monkey @ jpobst . com >
Merge pull request from ch5oh / master
2004 - 04 - 18 Sebastien Pouliot < sebastien @ ximian . com >
Test for .
[ sgen ] Remove CARDTABLE_STATS
2007 - 11 - 05 Zoltan Varga < vargaz @ gmail . com >
2008 - 05 - 28 Everaldo Canuto < ecanuto @ novell . com >
[ llvm ] Add mono_llvm_set_call_notail ( ) helper function , not yet used .
2008 - 02 - 25 Marek Safar < marek . safar @ gmail . com >
* SortedDictionary . cs ( ICollection . Contains ) : Use EqualityComparer
2003 - 11 - 10 Zoltan Varga < vargaz @ freemail . hu >
2006 - 10 - 26 Zoltan Varga < vargaz @ gmail . com >
[ profiler ] Clarify docs for mono_profiler_set_sample_mode ( ) .
we pass those
[ runtime ] Remove an obsolete option from - - help = devel .
1 . 0 build fix
[ runtime ] Add a _checked version for mono_class_from_name ( ) .
002 - 08 - 04 Ville Palo < vi64pa @ kolumbus . fi >
[ ci ] Remove confusing commented out code
cleanup
[ netcore ] Generate overwriting EMSCRIPTEN_SDK_DIR . Add a runtime build with threads support if ENABLE_WASM_THREADS is defined in Make . ( )
2002 - 06 - 14 Gaurav Vaish < gvaish @ iitk . ac . in >
[ profiler ] Fix the EqualityComparer . get_Default ( ) optimization for now , it seems to cause failures in the corlib tests on the ios simulator . ( )
Revert " [ sgen ] Remove sgen specific suspend code "
Add DecimalFormatterTest . cs from Patrick Kalkman
2007 - 11 - 22 Marek Safar < marek . safar @ gmail . com >
* BuildTask . cs : Instead of using
2004 - 07 - 15 Zoltan Varga < vargaz @ freemail . hu >
eh , open the floodgates . stop suppressing these warnings .
2003 - 11 - 14 Dietmar Maurer < dietmar @ ximian . com >
2001 - 12 - 26 Ravi Pratap < ravi @ ximian . com >
Bump roslyn ( to include more files )
fix ToDateTime ( XmlDateTimeSerializationMode . Unspecified ) / ToString ( DateTime ) for local time
[ jit ] Remove the support for MONO_ARCH_NOTIFY_PENDING_EXC , it only works on amd64 , and its not worth the complexity .
[ System . Net . Http ] Add IVT to Xamarin . * assemblies and update Mac
[ interp ] Remove virtual call duplicated code
Fix typo in EqualStringArrays ( dotnet / coreclr )
Up
Tue Aug 1 16 : 24 : 22 CEST 2002 Paolo Molaro < lupus @ ximian . com >
Update man page
[ System . Data . SqlClient ] Remove unused icalls .
some warning silence
2006 - 04 - 03 Sebastien Pouliot < sebastien @ ximian . com >
2004 - 12 - 21 Zoltan Varga < vargaz @ freemail . hu >
2008 - 07 - 18 Mark Probst < mark . probst @ gmail . com >
add reference file
2002 - 12 - 19 Duncan Mak < duncan @ ximian . com >
* * * empty log message * * *
New tests .
Remove redundant
2006 - 04 - 06 Gonzalo Paniagua Javier < gonzalo @ ximian . com >
* EventLog . cs : Minor fixes .
2009 - 02 - 13 Zoltan Varga < vargaz @ gmail . com >
2008 - 02 - 10 Miguel de Icaza < miguel @ novell . com >
ietmar to the changes
svn path = / trunk / mcs / ; revision = 51213
update csproj .
2004 - 02 - 10 Sebastien Pouliot < sebastien @ ximian . com >
* System . Web20 . csproj , System . Configuration20 . vmwcsproj : removed OrdinalComparer . cs
2005 - 02 - 14 Lluis Sanchez Gual < lluis @ novell . com >
Merge pull request from robwilkens / master
[ sdks ] Ensure gensources . cs is included in tarballs ( )
[ runtime ] mono_optimize_branches no longer breaks with OP_IL_SEQ_POINT .
* cs1502 . cs , HtmlElement . cs , HtmlHistory . cs , WebBrowser . cs :
[ crash ] Enable merp invocation ( )
Fixity fix
add MemberBind tests
2004 - 12 - 15 Sebastien Pouliot < sebastien @ ximian . com >
Mon Dec 13 17 : 50 : 19 CET 2002 Paolo Molaro < lupus @ ximian . com >
[ ci ] Change cominterop_raise_hr_exception to cominterop_set_hr_error
Removed hard coded timeout for SslStreamBase .
2006 - 05 - 03 Atsushi Enomoto < atsushi @ ximian . com >
* TcpServerChannel . cs : Changed default formatter sink to BinaryFormatterSink
PR feedback : GetElement ( 0 ) - > Sse2 . StoreLow
2002 - 07 - 22 Sebastien Pouliot < spouliot @ videotron . ca >
2004 - 05 - 02 Martin Baulig < martin @ ximian . com >
Merge pull request from monojenkins / sync - split
PR feedback : GetElement ( 0 ) - > Sse2 . StoreLow
new members to the graphic examples
Redirect right after NTLM authentication works now
Added support for UriComponents . SerializationInfoString . Fixes .
Add DecimalFormatterTest . cs to a test .
* XmlValidatingReader . cs : Removed the last AttributeSection class and
* Thread . cs ( MethodDef . CreateSignature ) : Use " ` n " only for generic
[ runtime ] Make sure MONO_CONTEXT_GET_CURRENT ( ) sets sp to an aligned value on watchos .
* System . Web . Services . dll . sources : Added new files :
2008 - 06 - 26 Atsushi Enomoto < atsushi @ ximian . com >
2008 - 04 - 13 Andreas Nahr < ClassDevelopment @ A - SoftTech . com >
[ jit ] Add a new ICALL_ADDR_CALL patch type which is only used on calls , since ICALL_ADDR might be used without a call .
Merge pull request from akoeplinger / fix - processtest - race
[ aot ] Simplify EmptyArray < enum > : . cctor ( ) instances since mcs now generates accesses to EmptyArray < enum > . Value for new enum [ 0 ] expressions .
svn path = / trunk / mcs / ; revision = 51213
refactoring
2010 - 06 - 16 Zoltan Varga < vargaz @ gmail . com >
2010 - 04 - 14 Carlos Alberto Cortez < calberto . cortez @ gmail . com >
2004 - 09 - 23 Zoltan Varga < vargaz @ freemail . hu >
[ dist ] Ensure gensources . cs is included in tarballs ( )
* ILParser . jay : Fix casting for data type constants .
2002 - 04 - 28 Miguel de Icaza < miguel @ ximian . com >
Merge pull request from kasthack / patch - 1
2003 - 09 - 25 Carlos Guzm � n � lvarez < carlosga @ telefonica . net >
2006 - 10 - 18 Zoltan Varga < vargaz @ gmail . com >
2006 - 02 - 25 Chris Toshok < toshok @ ximian . com >
2005 - 04 - 02 Atsushi Enomoto < atsushi @ ximian . com >
[ System . Private . CoreLib ] Add Nullable runtime support
* System . Web20 . csproj : added CompilationSection . cs
2006 - 03 - 15 Chris Toshok < toshok @ ximian . com >
2004 - 09 - 13 Atsushi Enomoto < atsushi @ ximian . com >
A test for bug
Fixed System . Net . Http . Headers
2004 - 03 - 11 Marek Safar < marek . safar @ seznam . cz >
2009 - 05 - 18 Zoltan Varga < vargaz @ gmail . com >
2001 - 09 - 28 Ravi Pratap < ravi @ ximian . com >
2005 - 05 - 29 Atsushi Enomoto < atsushi @ ximian . com >
* Wizard . cs : implemented SkipLinkText , TagKey
2008 - 01 - 29 Jonathan Pobst < monkey @ jpobst . com >
2004 - 09 - 14 Atsushi Enomoto < atsushi @ ximian . com >
2005 - 04 - 04 Gonzalo Paniagua Javier < gonzalo @ ximian . com >
2003 - 03 - 10 Gonzalo Paniagua Javier < gonzalo @ ximian . com >
Fix
2008 - 03 - 08 Zoltan Varga < vargaz @ gmail . com >
2004 - 07 - 07 Atsushi Enomoto < atsushi @ ximian . com >
[ sgen ] Fix a regression caused by 07fc727b8698a100aeb214ae3f7eb1cc3f5bab20 . Fixes the windows build .
DataRelationCollection . cs : Suppress finalization in Dispose ( ) method .
2009 - 09 - 24 Zoltan Varga < vargaz @ gmail . com >
[ xbuild ] Add a Microsoft . Portable . CSharp . targets to the list .
2006 - 04 - 07 Atsushi Enomoto < atsushi @ ximian . com >
Don', "'", 't add Array : : BinarySearch dependency for unreachable code
2008 - 07 - 06 Marek Habersack < mhabersack @ novell . com >
2008 - 05 - 22 Zoltan Varga < vargaz @ gmail . com >
2008 - 04 - 02 Ivan N . Zlatev < contact @ i - nz . net >
2002 - 05 - 05 Dietmar Maurer < dietmar @ ximian . com >
2002 - 09 - 13 Rafael Teixeira < rafaelteixeirabr @ hotmail . com >
Invert the condition
Update man page
Merge pull request from StephenMcConnel / bug - 24109
2008 - 05 - 25 Everaldo Canuto < ecanuto @ novell . com >
* * * empty log message * * *
2005 - 05 - 14 Peter Bartok < pbartok @ novell . com >
Add new test
be the Makefile
* DataColumnCollectionTest . cs : Added deprecation message .
[ runtime ] Enable application to register path for MERP restart
Merge pull request from lateralusX / lateralusX / fix - win - fullaot - runtime - tests - toggleref
2005 - 09 - 12 Zoltan Varga < vargaz @ gmail . com >
[ wasm ] [ debugger ] Clear valuetypes cache on Debugger . resume ( )
2003 - 05 - 21 Pedro Mart � nez Juli � < yoros @ wanadoo . es >
Little fix .
2007 - 09 - 08 Atsushi Enomoto < atsushi @ ximian . com >
2003 - 07 - 10 Sebastien Pouliot < spouliot @ videotron . ca >
[ configure ] Don', "'", 't show the deprecation notice for minimal or quiet logging
[ wasm ] [ debugger ] Hash messages and sessions directly ( )
[ corlib ] First from reference sources
Fixed typo in readme . md
2006 - 05 - 08 Chris Toshok < toshok @ ximian . com >
SortedSet : Complete add tests
[ System . EnterpriseServices ] Rename method parameters to match . NET contract
* WebConfigurationSettings . cs : Added deprecation message .
2010 - 04 - 26 Miguel de Icaza < miguel @ novell . com >
2008 - 05 - 06 Marek Safar < marek . safar @ gmail . com >
2005 - 03 - 10 Peter Bartok < pbartok @ novell . com >
[ System ] Add IPv6 support to FtpWebRequest
[ gensources ] : Add platform / project logic to tests .
Initial PrintDialog implementation of a bit
[ llvmonly ] Fix the build .
Bump to mono / corefx @ e79cf5b
Merge pull request from monojenkins / update - csprojs
add VS2005 repository
fix typo
2008 - 11 - 06 Miguel de Icaza < miguel @ novell . com >
* PEAPI . cs : Do not add file extensions to output names
- Fixed classes
[ corlib ] Restores ConcurrentDictionary tests on PRs
Bug improvement
2004 - 09 - 26 Jordi Mas i Hernandez < jordi @ ximian . com >
2004 - 07 - 12 Martin Baulig < martin @ ximian . com >
2008 - 12 - 24 Jb Evain < jbevain @ novell . com >
2009 - 05 - 26 Zoltan Varga < vargaz @ gmail . com >
2003 - 10 - 25 Sebastien Pouliot < spouliot @ videotron . ca >
* SizeGrip . cs : New internal class that is used as a sizing
2003 - 09 - 20 < cesar @ ciencias . unam . mx >
2007 - 03 - 22 Dick Porter < dick @ ximian . com >
[ Fixes ]
2003 - 02 - 18 Dick Porter < dick @ ximian . com >
[ coop ] Add a handle frame in construct_culture ( )
2004 - 03 - 10 Sebastien Pouliot < sebastien @ ximian . com >
Merge pull request from zevane / patch - 1
Fix a warning .
2009 - 05 - 09 Zoltan Varga < vargaz @ gmail . com >
2004 - 11 - 04 Martin Baulig < martin @ ximian . com >
Dear God , make me recompile my code after removing debug * before * I check in
2009 - 04 - 16 Rodrigo Kumpera < rkumpera @ novell . com >
2008 - 03 - 28 Zoltan Varga < vargaz @ gmail . com >
Fri Sep 28 19 : 26 : 51 CEST 2003 Paolo Molaro < lupus @ ximian . com >
Fix the build .
2006 - 09 - 06 Jb Evain < jbevain @ gmail . com >
2006 - 03 - 09 Sebastien Pouliot < sebastien @ ximian . com >
2006 - 03 - 24 Peter Dennis Bartok < pbartok @ novell . com >
more friendlyification
2001 - 07 - 04 Miguel de Icaza < miguel @ ximian . com >
* mono - api - diff . cs : Fixed line endings .
[ corlib ] First specific culture returned by enumerator was not fully initialized . Fixes
2002 - 11 - 13 Gonzalo Paniagua Javier < gonzalo @ ximian . com >
2004 - 07 - 22 Zoltan Varga < vargaz @ freemail . hu >
[ profiler ] Add new links
[ runtime ] Avoid allocating interface offsets multiple times . Fixes
2002 - 03 - 16 DennisHayes < dennish @ raytek . com >
2002 - 07 - 12 Nick Drochak < ndrochak @ gol . com >
Fixed bug
Increase tailcall instruction length on ppc64
2003 - 06 - 27 Zoltan Varga < vargaz @ freemail . hu >
2007 - 09 - 12 Jonathan Pobst < monkey @ jpobst . com >
Fix leaks in mono_lookup_internal_call_full_with_flags , for lack of destructors . ( )
[ bcl ] Fix the build .
2007 - 06 - 26 Jonathan Pobst < monkey @ jpobst . com >
2001 - 11 - 17 Ravi Pratap < ravi @ ximian . com >
2009 - 06 - 16 Atsushi Enomoto < atsushi @ ximian . com >
[ System ] EndRead now throws WebException on abort .
* SqlCommand . cs : Fixed param name in ArgumentException for negative
Add ExceptionDispatchInfo . SetCurrentStackTrace ( dotnet / coreclr )
[ xbuild ] Fix Tooltask . ToolPath behaviour to match . NET
2004 - 11 - 25 Cesar Lopez Nataren < cesar @ ciencias . unam . mx >
[ sgen ] GrayQueueEntry struct instead of just char * .
[ Microsoft . Build . BuildEngine ] Update tests for Android ( )
Fix handle leak in cominterop ( )
Merge pull request from ztzg / sql - server - bulk - insert
Thu Mar 16 17 : 43 : 51 CET 2006 Paolo Molaro < lupus @ ximian . com >
[ sgen ] Track down promoted size also with parallel minors
Update CVS policies
2008 - 08 - 28 Atsushi Enomoto < atsushi @ ximian . com >
2010 - 02 - 31 Jb Evain < jbevain @ novell . com >
2006 - 04 - 18 Senganal T < tsenganal @ novell . com >
fix ToDateTime ( XmlDateTimeSerializationMode . Unspecified ) / ToString ( DateTime ) for local time
[ Mono . Debugger . Soft ] Catching AssemblyMirror in AppDomainMirror and TypeMirror in AssemblyMirror
* ScriptCompilerInfo . cs : Calculate the maximum and change vars
Fri Jul 24 12 : 17 : 49 CEST 2004 Paolo Molaro < lupus @ ximian . com >
2009 - 04 - 30 Jb Evain < jbevain @ novell . com >
2004 - 09 - 28 Cesar Lopez Nataren < cesar @ ciencias . unam . mx >
First steps towards a new X event loop that manages to not crash . TODO : Add poll and cleanup timers
Added support for UriComponents . SerializationInfoString . Fixes .
2003 - 02 - 13 Gonzalo Paniagua Javier < gonzalo @ ximian . com >
In metadata :
Fix typo
* mono - api - diff . cs : Fixed line endings .
2006 - 07 - 30 Zoltan Varga < vargaz @ gmail . com >
New tests .
[ Coop ] ves_icall_RuntimeFieldInfo_GetRawConstantValue . ( )
2006 - 10 - 30 Carlos Alberto Cortez < calberto . cortez @ gmail . com >
2005 - 10 - 05 Sebastien Pouliot < sebastien @ ximian . com >
[ sgen ] Heavy statistics counters for scanned object descriptors .
2003 - 07 - 25 Sebastien Pouliot < spouliot @ videotron . ca >
Revert 73804
2008 - 03 - 24 Rodrigo Kumpera < rkumpera @ novell . com >
Lazy reports
[ ci ] Remove use of w32handle for File , Console , Pipe and Socket ( )
2009 - 10 - 03 Jb Evain < jbevain @ novell . com >
2002 - 12 - 08 Miguel de Icaza < miguel @ ximian . com >
2010 - 03 - 05 Zoltan Varga < vargaz @ gmail . com >
Fix
eh , open the floodgates . stop suppressing these warnings .
Add include for fd_set
Little fix to make this compile with gmcs .
Merge pull request from zevane / patch - 1
[ bcl ] Add regression test for bxc .
2009 - 01 - 15 Marek Habersack < mhabersack @ novell . com >
- Another one missed
Fix a warning .
Fixed inacurrate
2008 - 04 - 17 Gonzalo Paniagua Javier < gonzalo @ novell . com >
2008 - 05 - 27 Geoff Norton < gnorton @ novell . com >
[ arm ] Fix the build .
* MdiFormTest . cs : Switch to using Thread . CurrentCulture instead of
2005 - 04 - 22 Marek Safar < marek . safar @ seznam . cz >
[ sgen ] GrayQueueEntry struct instead of just char * .
[ jit ] Fix the check for calling mono_get_vtable_var ( ) , check for', "'", 'cfg - > method', "'", 'instead of', "'", 'method', "'", ', the latter is the inlined method . ( )
2009 - 06 - 17 Marek Habersack < mhabersack @ novell . com >
2009 - 09 - 12 Atsushi Enomoto < atsushi @ ximian . com >
[ csproj ] Update project files
[ jit ] Fix an unreachable code part in mini_emit_memset ( ) . Fixes . Thanks to Bill Seurer for spotting it .
* * * empty log message * * *
2002 - 04 - 28 Ravi Pratap < ravi @ ximian . com >
[ msvc ] Update csproj files ( )
passing those
Thu May 16 16 : 28 : 43 CEST 2010 Paolo Molaro < lupus @ ximian . com >
2008 - 05 - 14 Marek Safar < marek . safar @ gmail . com >
* Marshal . cs : Fixed Height property implementation and implemented
Update the build .
* Makefile : Add gacutil and delegates .
[ System . Net . Http ] Incomplete byte ranges parsing . Fixes
[ mcs ] Remove not implemented from ambiguous static type lookup . Fixes
[ interp ] Remove virtual call duplicated code
Moving from monodoc / scripts to mono / scripts . . .
2004 - 11 - 08 Marek Safar < marek . safar @ seznam . cz >
2005 - 12 - 01 Joerg Rosenkranz < joergr @ voelcker . com >
2001 - 10 - 28 Miguel de Icaza < miguel @ ximian . com >
2005 - 12 - 22 Ben Maurer < bmaurer @ ximian . com >
Fri Nov 14 16 : 28 : 30 CET 2001 Paolo Molaro < lupus @ ximian . com >
[ sdks ] Add a _checked test around mono_class_from_name ( ) .
[ WindowsBase ] Add CreateTypeInfo method
2010 - 05 - 02 Geoff Norton < gnorton @ novell . com >
[ System . Windows . Forms ] Use local loop in 8080 to 30158
Bug fix
Restore Previous ExecutionContext Capture Behavior ( dotnet / coreclr )
[ System ] Add IPv6 support to FtpWebRequest
TARGET_JVM
* * * * Merged r41636 from MCS * * * *
2002 - 10 - 13 Gonzalo Paniagua Javier < gonzalo @ ximian . com >
2006 - 11 - 04 John Luke < jluke @ cfl . rr . com >
2005 - 05 - 10 Alp Toker < alp @ atoker . com >
2005 - 01 - 06 Marek Safar < marek . safar @ seznam . cz >
Changed splitter so it can be used by direct types
New tests .
2002 - 04 - 23 Tim Coleman < tim @ timcoleman . com >
2005 - 09 - 29 Martin Baulig < martin @ ximian . com >
Merge pull request from lateralusX / lateralusX / add - waitforkeypress - test - project
2003 - 01 - 18 Sebastien Pouliot < spouliot @ videotron . ca >
2005 - 02 - 07 Atsushi Enomoto < atsushi @ ximian . com >
2006 - 02 - 24 Jb Evain < jbevain @ gmail . com >
2005 - 10 - 30 Miguel de Icaza < miguel @ novell . com >
2002 - 05 - 16 Nick Drochak < ndrochak @ gol . com >
2007 - 02 - 10 Atsushi Enomoto < atsushi @ ximian . com >
2007 - 10 - 06 Jonathan Pobst < monkey @ jpobst . com >
2002 - 11 - 22 Tim Coleman < tim @ timcoleman . com >
2002 - 04 - 24 Miguel de Icaza < miguel @ ximian . com >
[ io - layer ] Extract _wapi_search_handle_namespace from handles . c
2008 - 01 - 14 Martin Baulig < martin @ ximian . com >
2007 - 09 - 14 Rodrigo Kumpera < rkumpera @ novell . com >
2009 - 06 - 10 Atsushi Enomoto < atsushi @ ximian . com >
[ llvm ] Fix the check for calling mono_get_vtable_var ( ) , check for', "'", 'cfg - > method', "'", 'instead of', "'", 'method', "'", ', the latter is the inlined method . ( )
Fix bug , fix build
2009 - 05 - 24 Marek Habersack < mhabersack @ novell . com >
2010 - 04 - 23 Atsushi Enomoto < atsushi @ ximian . com >
* statement . cs ( Statement . Clone ) : Move special case code to Block . CloneTo .
Update test
Changes sourceStream . Read to read in a while loop
2009 - 03 - 04 Rodrigo Kumpera < rkumpera @ novell . com >
2009 - 09 - 20 Marek Habersack < mhabersack @ novell . com >
2008 - 11 - 22 Ivan N . Zlatev < contact @ i - nz . net >
removed test file
2006 - 03 - 13 Mike Kestner < mkestner @ novell . com >
[ xbuild ] Add CreateTypeInfo method
2006 - 07 - 01 Chris Toshok < toshok @ ximian . com >
[ coop ] Switch to GC safe around pthread_join
2003 - 10 - 16 Zoltan Varga < vargaz @ freemail . hu >
2009 - 07 - 01 Atsushi Enomoto < atsushi @ ximian . com >
some people claim they have old SYSMONO that cannot compile this source and demand to " fix " this code .
[ profiler ] Clarify docs for mono_profiler_set_sample_mode ( ) .
2009 - 03 - 04 Marek Safar < marek . safar @ gmail . com >
PR : And then rename the image back .
2006 - 01 - 24 Cesar Lopez Nataren < cnataren @ novell . com >
Update
2008 - 08 - 26 Marek Safar < marek . safar @ gmail . com >
[ netcore ] Minor fixes around System . Reflection ( )
2010 - 05 - 26 Mark Probst < mark . probst @ gmail . com >
2010 - 10 - 19 Veerapuram Varadhan < vvaradhan @ novell . com >
2002 - 07 - 11 Nick Drochak < ndrochak @ gol . com >
* XmlTypeMapping . cs : Modified GetRealTypeMap to return current map if map
Merge pull request from marek - safar / build - fixes
[ sgen ] Heavy major sections before / after sweep correctly .
Revert " [ runtime ] Fix the computation of token sizes in ppdb files , the PDB stream has an additional table which contains the token sizes for the metadata tables referenced by the ppdb tables . ( ) "
2008 - 08 - 24 Marek Safar < marek . safar @ gmail . com >
Thu Nov 5 16 : 28 : 58 CET 2006 Paolo Molaro < lupus @ ximian . com >
update
[ wasm ] Fix CoreXml . ToolPath ( )
* DataGrid . cs : call SetDataSource instead of CalcGridAreas in
2006 - 06 - 19 Carlos Alberto Cortez < calberto . cortez @ gmail . com >
2009 - 07 - 11 Rodrigo Kumpera < rkumpera @ novell . com >
TARGET_JVM : add msbuild project file
2008 - 12 - 21 Rodrigo Kumpera < rkumpera @ novell . com >
2006 - 02 - 13 Sebastien Pouliot < sebastien @ ximian . com >
Fix watchos tests for system . data ( )
* * * empty log message * * *
[ wasm ] Adjust packager to better work with our sample . Update the sample to use the binding engine and , finally , update the README with this all .
Nullable : Int32 ( dotnet / coreclr )
[ runtime ] Make MONO_CONTEXT_GET_CURRENT ( ) quiet . ( )
2004 - 03 - 16 Ben Maurer < bmaurer @ users . sourceforge . net >
Don', "'", 't Sleep ( 1 ) in some spin - wait loops ( dotnet / coreclr )
2007 - 01 - 29 Jonathan Pobst < monkey @ jpobst . com >
2007 - 05 - 25 Sebastien Pouliot < sebastien @ ximian . com >
[ bcl ] Add the Microsoft . Portable . CSharp . dll to the monotouch build .
Makefile .
[ monkeydoc ] Avoid using BinarySerialization
[ threadpool - ms ] Fix incorrect use of mono_rand .
2008 - 05 - 07 Ivan N . Zlatev < contact @ i - nz . net >
2005 - 09 - 04 Atsushi Enomoto < atsushi @ ximian . com >
2005 - 06 - 10 Maverson Eduardo Schulze Rosa < maverson @ gmail . com >
[ jit ] Enable LLVM mode . "
Fix optimizatons in preperation for Jackson', "'", 's jay patch
Remove some unused icalls .
[ Coop ] ves_icall_RuntimeFieldInfo_GetRawConstantValue . ( )
Fix two new nullable warnings with latest compiler
* PEAPI . cs : Removed the redundant AttributeSection class and
[ sgen ] Fix build on C99 .
2007 - 01 - 15 Dick Porter < dick @ ximian . com >
New test .
[ monkeydoc ] Steps fix for Int32Rect , Point , Rect , Size , Vector , Matrix ( )
[ runtime ] mono_optimize_branches no longer breaks with OP_IL_SEQ_POINT .
2005 - 06 - 05 Gonzalo Paniagua Javier < gonzalo @ ximian . com >
2003 - 11 - 04 Gonzalo Paniagua Javier < gonzalo @ ximian . com >
[ bcl ] Fix the build .
Warnings cleanup
Use _vscprintf ( ) for mingw - w64 compilers
2003 - 12 - 15 Atsushi Enomoto < ginga @ kit . hi - ho . ne . jp >
2005 - 03 - 31 Lluis Sanchez Gual < lluis @ novell . com >
2006 - 01 - 31 Sebastien Pouliot < sebastien @ ximian . com >
[ corlib ] Rename method parameters to macos .
2004 - 03 - 20 Zoltan Varga < vargaz @ freemail . hu >
[ sgen ] Don', "'", 't wait for sweep finish when allocating major object
[ jit ] Add a new ICALL_ADDR_CALL patch type which is only used on calls , since ICALL_ADDR might be used without a call .
2002 - 10 - 21 Rachel Hestilow < hestilow @ ximian . com >
Revert SVN r53054
2008 - 03 - 28 Carlos Alberto Cortez < calberto . cortez @ gmail . com >
Add DecimalFormatterTest . cs from Patrick Kalkman
* WebConfigurationSettings . cs : Use new culture info method for constructing the
Merge pull request from lambdageek / bug - 49686
Add new test
[ corlib ] Reset List enumerator current field . Fixes
[ jit ] Remove some dead code / includes .
ChangeLog : Updated ChangeLog .
* RemotingServices . cs : Temp source files should have extension " . vb " .
2007 - 05 - 11 Jonathan Pobst < monkey @ jpobst . com >
New test .
Merge pull request from akoeplinger / fix - processtest - race
[ mcs ] Reset doc reader state after expression bodied property . Fixes
2005 - 01 - 14 Zoltan Varga < vargaz @ freemail . hu >
remove classref hack
Standardize file - endings . Set svn : eol - style = native on LF - terminated files .
[ 642437 ] Nested types have to check type arguments only and not full class
2003 - 07 - 24 Zoltan Varga < vargaz @ freemail . hu >
[ runtime ] Include sys / time . h to get struct timeval definition .
2009 - 09 - 27 Sebastien Pouliot < sebastien @ ximian . com >
* TreeView . cs : Use dashed lines to connect nodes . Use the
fixes test .
2005 - 04 - 15 Cesar Lopez Nataren < cesar @ ciencias . unam . mx >
2005 - 06 - 14 Zoltan Varga < vargaz @ freemail . hu >
2003 - 02 - 20 Atsushi Enomoto < ginga @ kit . hi - ho . ne . jp >
2004 - 01 - 02 Miguel de Icaza < miguel @ ximian . com >
Fix a warning .
2006 - 02 - 19 Alexander Olk < alex . olk @ googlemail . com >
[ bcl ] Update csproj files
2006 - 12 - 19 Marek Safar < marek . safar @ seznam . cz >
2002 - 01 - 16 Nick Drochak < ndrochak @ gol . com >
fix ToDateTime ( XmlDateTimeSerializationMode . Unspecified ) / ToString ( DateTime ) for local time
* NumberFormatter . cs : Fixed rounding for float and the string output now includes all the precision ( not counting preceding zeros ) . This fix the DecimalTest . TestConstructSingleRounding_NotWorking test cases .
2006 - 12 - 01 Zoltan Varga < vargaz @ gmail . com >
* driver . cs ( CodeGeneratorFromTypeTest_Class )
2002 - 09 - 28 Gonzalo Paniagua Javier < gonzalo @ ximian . com >
Add span - based CopyTo and CopyToAsync methods ( dotnet / coreclr )
* HttpRequest . cs : Set the cache', "'", 's duration .
2006 - 10 - 13 Zoltan Varga < vargaz @ gmail . com >
2007 - 09 - 29 Jonathan Pobst < monkey @ jpobst . com >
New test .
[ coop ] Add a handle frame in construct_culture ( )
* FieldDef . cs : Set the underlying type for bytearray data constants . This
2007 - 07 - 06 Nagappan A < anagappan @ novell . com >
2007 - 09 - 04 Marek Safar < marek . safar @ gmail . com >
[ System . Type ] Changed test port from 8080 to 30158
Tue Sep 16 16 : 03 : 14 CEST 2002 Paolo Molaro < lupus @ ximian . com >
Bump roslyn ( to include more files )
use 32bpp bitmaps internally
[ mcs ] eval . cs : add ValueModificationHandler delegate to be more clear
2006 - 07 - 31 Senganal T < tsenganal @ novell . com >
2002 - 06 - 10 Patrik Torstensson
Mon Aug 4 12 : 12 : 26 CEST 2003 Paolo Molaro < lupus @ ximian . com >
2009 - 05 - 10 Ivan N . Zlatev < contact @ i - nz . net >
2010 - 05 - 19 Marek Safar < marek . safar @ gmail . com >
* * * empty log message * * *
[ runtime ] Work around an ios9 problem where dlopen ( ) cant', "'", 'find system libraries specified without paths .
[ 642437 ] Nested types have to check type arguments only and not full class
2002 - 09 - 12 Daniel Morgan < danmorg @ sc . rr . com >
2008 - 01 - 21 Zoltan Varga < vargaz @ gmail . com >
Update
[ ci ] Remove - f option from genhtml during code coverage
2006 - 10 - 01 Rolf Bjarne Kvinge < RKvinge @ novell . com >
[ Coop ] Convert performance counters and performance counter categories . ( )
2002 - 12 - 14 Gonzalo Paniagua Javier < gonzalo @ ximian . com >
refactoring
2001 - 11 - 21 Rachel Hestilow < hestilow @ ximian . com >
2002 - 07 - 12 Rachel Hestilow < hestilow @ ximian . com >
2002 - 06 - 14 Jeffrey Stedfast < fejj @ ximian . com >
Unbreak the build
New test .
bootstrap fix
2003 - 07 - 30 Todd Berman < tberman @ gentoo . org >
column name and ordinal fix . . . tested on 10 . 1
Merge pull request from esdrubal / xmlgenerator
* ConstraintCollectionTest2 . cs
Fix a warning .
fix build error
Updated .
2010 - 06 - 10 Sebastien Pouliot < sebastien @ ximian . com >
[ interp ] Enable inlining of object constructors ( )
2005 - 08 - 29 Atsushi Enomoto < atsushi @ ximian . com >
Merge pull request from ntherning / fix - broken - windows - build
Move Interlocked
[ jit ] Simplify the seq point code used for soft breakpoints .
2009 - 02 - 25 Marek Habersack < mhabersack @ novell . com >
2008 - 05 - 21 Geoff Norton < gnorton @ novell . com >
2003 - 09 - 26 Ben Maurer < bmaurer @ users . sourceforge . net >
Fix
2003 - 02 - 14 Zoltan Varga < vargaz @ freemail . hu >
[ runtime ] Fix build when using musl libc .
2005 - 07 - 04 Sebastien Pouliot < sebastien @ ximian . com >
[ runtime ] Fix a warning .
Merge pull request from lateralusX / lateralusX / fix - simd - llvm - issue
[ sgen ] Make unnecessary stack check into assertion .
Thu Mar 16 19 : 43 : 53 CET 2008 Paolo Molaro < lupus @ ximian . com >
2004 - 06 - 25 Sebastien Pouliot < sebastien @ ximian . com >
Only print the', "'", 'Assuming PLATFORM messages when running make with V = 1 .
Pending changes
from head
2007 - 10 - 31 Atsushi Enomoto < atsushi @ ximian . com >
* BuildTask . cs ( Execute ) : Use task_logger only after the task
[ ppdb ] Remove unused icalls for Environment properties ( )
2004 - 06 - 11 Gonzalo Paniagua Javier < gonzalo @ ximian . com >
[ checked ] Eliminate warning about redefinition of GC_WIN32_THREADS
Add StatusBarDrawItem stuff to build
Fix the build .
* XmlTypeMapping . cs : Modified GetRealTypeMap to return current map if map
2009 - 11 - 09 Sebastien Pouliot < sebastien @ ximian . com >
2005 - 07 - 31 Miguel de Icaza < miguel @ novell . com >
2003 - 11 - 05 Zoltan Varga < vargaz @ freemail . hu >
2005 - 09 - 06 Atsushi Enomoto < atsushi @ ximian . com >
Ensure some of the tests
[ sdb ] Simplify the locking in set_breakpoint ( ) a bit .
[ 642437 ] Nested types have to check type arguments only and not full class
[ acceptance - tests ] Bump coreclr to pick up 3b2969dffe191b03f1ac85a9c1e62417c9b7833b :
Restore Previous ExecutionContext Capture Behavior ( dotnet / coreclr )
2008 - 10 - 14 Zoltan Varga < vargaz @ gmail . com >
2002 - 08 - 16 Miguel de Icaza < miguel @ ximian . com >
Cleanup
2006 - 04 - 30 Atsushi Enomoto < atsushi @ ximian . com >
2004 - 03 - 15 Ben Maurer < bmaurer @ users . sourceforge . net >
Add StatusBarDrawItem stuff to build
[ test ] Have mono_class_from_name with mono_class_load_from_name ( )
2008 - 10 - 19 Ivan N . Zlatev < contact @ i - nz . net >
* main . c ( dis_method_list ) : Oops . Check method_context for null .
2004 - 01 - 22 Sebastien Pouliot < sebastien @ ximian . com >
[ threadpool - io ] Remove sparse - array for poll backend
[ runtime ] Register the copying of EcmaDesc
Merge pull request from lateralusX / lateralusX / fix - win - fullaot - runtime - tests - toggleref
Fix annoying typos
Mon May 4 16 : 00 : 37 CEST 2008 Paolo Molaro < lupus @ ximian . com >
2007 - 04 - 27 Atsushi Enomoto < atsushi @ ximian . com >
2005 - 12 - 21 Martin Baulig < martin @ ximian . com >
Tue May 6 13 : 00 : 51 CEST 2004 Paolo Molaro < lupus @ ximian . com >
2007 - 02 - 13 Zoltan Varga < vargaz @ gmail . com >
Fix warning building . ( )
Merge pull request from akoeplinger / fix - processtest - race
another error message for default support
Add experimental files
2005 - 07 - 27 Zoltan Varga < vargaz @ freemail . hu >
Unixify
2002 - 03 - 21 Duncan Mak < duncan @ ximian . com >
Maintain max_len when emitting calls on ARM hard float .
* PlatformID . cs ( MethodDef . CreateSignature ) : Use " ` n " only for generic
New tests .
[ System ] Semaphore from referencesource
[ mini ] Remove CARDTABLE_STATS
Fixed SGen qsort code to compile with MSVC .
2009 - 04 - 03 Sebastien Pouliot < sebastien @ ximian . com >
we pass those
[ System ] Add IPv6 support to FtpWebRequest
removed test file
2006 - 02 - 22 Marek Sieradzki < marek . sieradzki @ gmail . com >
Add IBM . SetCurrentStackTrace ( dotnet / coreclr )
2007 - 04 - 31 Marek Habersack < mhabersack @ novell . com >
2005 - 07 - 20 Zoltan Varga < vargaz @ freemail . hu >
[ xbuild ] When parsing the ECMA key , don', "'", 't produce a public key . Fixes
2006 - 03 - 27 Zoltan Varga < vargaz @ gmail . com >
2004 - 03 - 31 Atsushi Enomoto < atsushi @ ximian . com >
Fix eglibc configure script on Haiku
2009 - 05 - 28 Zoltan Varga < vargaz @ gmail . com >
add test - coreclr - security till is fixed
* System . Configuration . dll . sources : added more stubs for classes
[ runtime ] Fix build when using musl libc .
2007 - 09 - 06 Marek Habersack < mhabersack @ novell . com >
[ mcs ] Filter out nested private types of forwarded types
[ jit ] Get rid of the intermediate OP_ . . CALLVIRT opcodes , use the _MEMBASE opcodes instead .
Add attributes and serializable attributes .
Update
2002 - 09 - 24 Dietmar Maurer < dietmar @ ximian . com >
[ 642437 ] Nested types have to check type arguments only and not full class
[ arm ] Fix a regression introduced by : ( )
Merge pull request from zevane / patch - 1
Finished changes
[ wasm ] [ debugger ] Hash messages and sessions directly ( )
[ w32handle ] Remove binary call duplicated code
more docu
2006 - 02 - 25 Carlos Alberto Cortez < calberto . cortez @ gmail . com >
[ System . XML ] Fix handling of global variables in XSLs ( )
2003 - 09 - 01 Dietmar Maurer < dietmar @ ximian . com >
2009 - 12 - 08 Rodrigo Kumpera < rkumpera @ novell . com >
* VBCodeProviderTest . cs : Use dashed lines to connect nodes . Use the
[ runtime ] Fix a warning .
2010 - 06 - 09 Zoltan Varga < vargaz @ gmail . com >
[ runtime ] Get rid of the create_delegate_trampoline_data ( ) helper function which only has one caller .
* * * empty log message * * *
[ sgen ] Add some explanatory comment .
2005 - 08 - 30 John Luke < john . luke @ gmail . com >
New test .
Fix build break in TimerQueue . Portable . cs
2005 - 07 - 02 Ben Maurer < bmaurer @ ximian . com >
add MemberBind tests
[ Tailcall ] Convert dynamic - image . c . ( )
2009 - 01 - 04 Zoltan Varga < vargaz @ gmail . com >
Changes sourceStream . Read to read in a while loop
2009 - 04 - 06 Bill Holmes < billholmes54 @ gmail . com >
2008 - 12 - 04 Mark Probst < mark . probst @ gmail . com >
2008 - 04 - 21 Gonzalo Paniagua Javier < gonzalo @ novell . com >
[ wasm ] [ debugger ] Clear valuetypes cache on Debugger . resume ( )
2005 - 12 - 29 Gonzalo Paniagua Javier < gonzalo @ ximian . com >
A test kumpera bug
* SerializationCodeGenerator . cs : Minor fixes .
2009 - 03 - 03 Atsushi Enomoto < atsushi @ ximian . com >
* SoapFormatter . cs : Only apply ComVisible attribute to FilterLevel
2007 - 11 - 20 Zoltan Varga < vargaz @ gmail . com >
2006 - 01 - 09 Chris Toshok < toshok @ ximian . com >
[ mcs ] Add missing legacy GetTitle method
2008 - 05 - 23 Zoltan Varga < vargaz @ gmail . com >
Initialization of default proxy added .
Pending changes
2009 - 07 - 31 Atsushi Enomoto < atsushi @ ximian . com >
[ amd64 ] Add definition of GSharedVtCallInfo to mini - amd64 . h
[ jit ] Get rid of the intermediate OP_ . . CALLVIRT opcodes , use the _MEMBASE opcodes instead .
Merge pull request from bassamtabbara / master
2007 - 11 - 05 Sebastien Pouliot < sebastien @ ximian . com >
[ System ] EndRead now throws WebException on abort .
2008 - 12 - 19 Marek Safar < marek . safar @ gmail . com >
2004 - 07 - 07 Geoff Norton < gnorton @ customerdna . com >
Mon Aug 30 17 : 47 : 15 CEST 2001 Paolo Molaro < lupus @ ximian . com >
[ threadpool - ms ] Fix incorrect use of mono_rand .
2002 - 10 - 09 Gaurav Vaish < gvaish_mono @ lycos . com >
2005 - 05 - 22 Sebastien Pouliot < sebastien @ ximian . com >
[ runtime ] Make mono_value_box use MonoError
2008 - 07 - 18 Sebastien Pouliot < sebastien @ ximian . com >
[ tests ] Xamarin . Android has the desktop behavior
[ System . EnterpriseServices ] Rename method parameters to match . NET contract
[ wasm ] [ debugger ] Add support for surfacing inherited members ( )
2002 - 02 - 14 Patrik Torstensson < patrik . torstensson @ labs2 . com >
2010 - 05 - 06 Jérémie Laval < jeremie . laval @ gmail . com >
Merge pull request from cmp - / remove - obsolete - stack - checks - win32
Detect inifinite generic recursion differently in the AOT compiler .
bump nunitlite
New tests .
Merge pull request from akoeplinger / fix - processtest - race
2008 - 10 - 06 Marek Safar < marek . safar @ gmail . com >
2004 - 05 - 22 Atsushi Enomoto < atsushi @ ximian . com >
2008 - 01 - 24 Jb Evain < jbevain @ novell . com >
Fix Debug . Assert in GenericDelegateCache ( dotnet / coreclr )
[ jit ] Transition the SIMD code to use the MONO_CPU_ . . . set of flags instead of the SIMD_VERSION_ . . . flags . ( )
* TreeView . cs : Use LostFocus instead of Leave for
[ eglib ] Add Mono . Cecil . * . Has * checks when comparing methods / types
2010 - 01 - 31 Marek Safar < marek . safar @ gmail . com >
2009 - 08 - 30 Marek Safar < marek . safar @ gmail . com >
2007 - 03 - 19 Jonathan Pobst < monkey @ jpobst . com >
[ jit ] Disable some cctors for inlined methods , the cctor might not have ran in AOT mode . ( )
New test .
Final changes
* ComboBoxTest . cs : Added DataBindingTest .
2004 - 09 - 04 Sebastien Pouliot < sebastien @ ximian . com >
Merge pull request from lambdageek / dev / monoerror - stringy
2010 - 11 - 11 Zoltan Varga < vargaz @ gmail . com >
2009 - 07 - 04 Miguel de Icaza < miguel @ novell . com >
[ bcl ] When finalizing a Process object , don', "'", 't touch the streams as they are finalized on their own . Fixes .
[ metadata ] Fix the safe foreach macro .
* SoapFormatter . cs : Only apply ComVisible attribute to FilterLevel
Put the release
Sat May 4 12 : 00 : 39 CEST 2002 Paolo Molaro < lupus @ ximian . com >
[ runtime ] Fix a warning .
Add class back
[ sgen ] New protocol entry for processing of individual reference .
2009 - 02 - 01 Atsushi Enomoto < atsushi @ ximian . com >
Add to the distribution
* ThemeWin32Classic . cs : When imagelists are used the image area is
2008 - 05 - 26 Jonathan Pobst < monkey @ jpobst . com >
Remove the monotouch directives from AttributeCollection . cs .
* PageCompiler . cs : Add Trace and TraceMode to framework initialize
Added a workaround for .
Format is now setting FormatFlags . HasUriCharactersToNormalize .
[ profiler ] Fix the EqualityComparer . get_Default ( ) optimization for now , it seems to cause failures in the corlib tests on the ios simulator . ( )
[ System ] Fix basic build
Change ToUpper / ToLower throws for overlapping buffer ( )
* main . c ( dis_method_list ) : Oops . Check method_context for null .
[ mini ] Fix CAS atomic ordering under LLVM .
2003 - 07 - 14 Dick Porter < dick @ ximian . com >
2008 - 05 - 14 Gonzalo Paniagua Javier < gonzalo @ novell . com >
2009 - 04 - 26 Ivan N . Zlatev < contact @ i - nz . net >
* HtmlDocument . cs , HtmlElement . cs , HtmlHistory . cs , WebBrowser . cs :
[ Coop ] ves_icall_RuntimeFieldInfo_GetRawConstantValue . ( )
2006 - 06 - 19 Peter Dennis Bartok < pbartok @ novell . com >
Boehm doesn', "'", 't need the pinning root for threads .
2007 - 08 - 29 Stephane Delcroix < sdelcroix @ novell . com >
2002 - 08 - 25 Martin Baulig < martin @ gnome . org >
cls happyfication
Merge pull request from akoeplinger / pupitetris - Bug - 35635
Merge pull request from akoeplinger / fix - processtest - race
Merge pull request from kumpera / wasm_fix_a_few_issues
2007 - 04 - 06 Adar Wesley < adarw @ mainsoft . com >
revert
DennisHayes < T > : Validate parameters in the public methods ( )
2004 - 05 - 15 Ben Maurer < bmaurer @ users . sourceforge . net >
New tests .
fix test
2005 - 09 - 10 Peter Dennis Bartok < pbartok @ novell . com >
* * * empty log message * * *
New test .
[ corlib ] Fix culture info error message . Fixes
Implemented NullTarget ( )
[ MonoError ] Use mono_error_set_not_implemented helper
2008 - 11 - 22 Zoltan Varga < vargaz @ gmail . com >
[ mcs ] Inflate anonymymous method with correct type arguments when async this requires hoisting in parent type . Fixes
* Tds . cs : New file - Represents a reference to a type in
2008 - 11 - 08 Zoltan Varga < vargaz @ gmail . com >
[ xbuild ] Add CreateTypeInfo method
2005 - 07 - 29 Ben Maurer < bmaurer @ ximian . com >
Merge pull request from monojenkins / update - csprojs
[ sgen ] Fix a toggleref array resize bug .
2001 - 11 - 24 Dan Lewis < dihlewis @ yahoo . co . uk >
2007 - 08 - 31 Jonathan Pobst < monkey @ jpobst . com >
2004 - 03 - 30 Gonzalo Paniagua Javier < gonzalo @ ximian . com >
2005 - 08 - 19 Lluis Sanchez Gual < lluis @ novell . com >
[ sgen ] GrayQueueEntry struct instead of just char * .
Clean was a duplicated error test with 368 , update to test something new
Merge pull request from esdrubal / utf8encoding
2007 - 01 - 29 Jonathan Pobst < monkey @ jpobst . com >
2002 - 03 - 14 Tim Coleman < tim @ timcoleman . com >
Fix parent instantantiation verification .
Fix typo ( for MOBILE ) that excluded System . Xml . Schema / Extensions . cs . Removed the condition since every profile we now support includes this [ ]
2010 - 03 - 04 Gonzalo Paniagua Javier < gonzalo @ novell . com >
2009 - 04 - 03 Zoltan Varga < vargaz @ gmail . com >
[ jit ] Remove the support for MONO_ARCH_NOTIFY_PENDING_EXC , it only works on amd64 , and its not worth the complexity .
Thu May 5 16 : 00 : 52 CEST 2002 Paolo Molaro < lupus @ ximian . com >
2005 - 12 - 02 Gonzalo Paniagua Javier < gonzalo @ ximian . com >
2009 - 07 - 19 Rodrigo Kumpera < rkumpera @ novell . com >
Implemented Encoder . cs .
Merge pull request from ch5oh / master
2005 - 10 - 05 Geoff Norton < gnorton @ customerdna . com >
Add new files
Revert " [ marshal ] delay dummy string allocation "
* inssel . c ( build_assembly_name ) : return FALSE if length of token is
Warnings cleanup
2009 - 04 - 14 Sebastien Pouliot < sebastien @ ximian . com >
Added new enumeration for TLS compression methods
2002 - 05 - 17 Tim Coleman < tim @ timcoleman . com >
[ jit ] Get rid of the create_delegate_trampoline_data ( ) helper function which only has one caller .
2009 - 11 - 24 Marek Habersack < mhabersack @ novell . com >
Add class ForestTrustDomainInformation
Nullable : Int32 ( dotnet / coreclr )
2010 - 04 - 09 Atsushi Enomoto < atsushi @ ximian . com >
2006 - 06 - 17 Martin Baulig < martin @ ximian . com >
Switch to precise module
2007 - 10 - 17 Atsushi Enomoto < atsushi @ ximian . com >
2008 - 05 - 05 Jonathan Pobst < monkey @ jpobst . com >
2007 - 02 - 12 Geoff Norton < gnorton @ novell . com >
build one
2005 - 04 - 26 Atsushi Enomoto < atsushi @ ximian . com >
This is not supported now
Fix boehm after eb1a7c159149745dfbc45ca9de81a7585b82b993 .
2005 - 09 - 28 Peter Dennis Bartok < pbartok @ novell . com >
2006 - 04 - 06 Gonzalo Paniagua Javier < gonzalo @ ximian . com >
2008 - 10 - 08 Marek Safar < marek . safar @ gmail . com >
2007 - 11 - 05 Zoltan Varga < vargaz @ gmail . com >
* BitVector32 . cs : Minor fixes .
Fix the System . Core build for the monotouch profile
Bump roslyn ( to include more files )
2005 - 07 - 11 Atsushi Enomoto < atsushi @ ximian . com >
Revert 90304 , it broke the build
[ coop ] Enter blocking mode before waiting on a condvar in the threadpool code .
[ wasm ] [ bcl ] [ zoneinfo ] Fix local zone info marshaling ( )
Merge pull request from tpokorra / TarballFixAvoidCDataTextReaderTests
2007 - 01 - 13 Jeffrey Stedfast < fejj @ novell . com >
* ScrollableControl . cs : Calculate the maximum and change vars
Merge pull request from monojenkins / bug - 49686
2008 - 04 - 31 Marek Safar < marek . safar @ gmail . com >
[ runtime ] Register get gsharedvt arg trampolines
[ Mono . Security ] : Add API to let MonoTlsProvider provider a custom system certificate validator .
Fix eglibc uninitialized script .
PR feedback : GetElement ( 0 ) - > Sse2 . StoreLow
Simplified DataSetTests from Mainsoft
Thu Aug 8 16 : 24 : 43 CEST 2005 Paolo Molaro < lupus @ ximian . com >
2006 - 10 - 02 Chris Toshok < toshok @ ximian . com >
[ runtime ] Fix a warning .
Don', "'", 't use MonoGenericParamInfo for ownerless generic params .
2004 - 09 - 07 Atsushi Enomoto < atsushi @ ximian . com >
2004 - 05 - 29 Dick Porter < dick @ ximian . com >
2004 - 03 - 13 Miguel de Icaza < miguel @ ximian . com >
Merge pull request from elijahtaylor / fixdyncode
2006 - 01 - 14 Zoltan Varga < vargaz @ gmail . com >
* SqlDataReader . cs : Only stop server if it was actually created .
Flush
[ runtime ] Remove some dead code and data .
2007 - 03 - 11 Everaldo Canuto < everaldo @ simios . org >
Wed May 13 13 : 30 : 51 CEST 2002 Paolo Molaro < lupus @ ximian . com >
2009 - 01 - 06 Atsushi Enomoto < atsushi @ ximian . com >
* MethodInstr . cs : callvirt operation is implicitly instance .
2007 - 12 - 16 Sebastien Pouliot < sebastien @ ximian . com >
2004 - 04 - 15 Umadevi S < sumadevi @ novell . com >
DIST_SUBDIRS is a thing , fixes parallelism broken by previous change
2007 - 04 - 03 Nagappan A < anagappan @ novell . com >
[ reflection ] Add definition of GSharedVtCallInfo to 30158
2007 - 04 - 02 Juraj Skripsky < js @ hotfeet . ch >
2009 - 01 - 28 Christian Hergert < chris @ dronelabs . com >
2002 - 02 - 23 Tim Coleman < tim @ timcoleman . com >
[ sgen ] Fix a toggleref array resize bug .
2008 - 01 - 29 Zoltan Varga < vargaz @ gmail . com >
[ mono ] Implement Arm intrinsics : ArmBase , Crc32 ( )
[ arm ] Fix a regression introduced by : ( )
[ netcore ] Update project files
2006 - 01 - 06 Marek Sieradzki < marek . sieradzki @ gmail . com >
2004 - 07 - 12 Dick Porter < dick @ ximian . com >
In Test / System . Data :
2005 - 12 - 01 Martin Baulig < martin @ ximian . com >
2004 - 05 - 13 Gonzalo Paniagua Javier < gonzalo @ ximian . com >
[ system ] Make ICommand 4 . 5 profile only
* XmlReflectionImporter . cs : Use dashed lines to connect nodes . Use the
[ arm64 ] Allocate thunk space for tail calls . Fixes .
2002 - 08 - 06 Rachel Hestilow < hestilow @ ximian . com >
[ interp ] Use frame_addr as base pointer in more places ( )
2006 - 09 - 26 Zoltan Varga < vargaz @ gmail . com >
Unbreak the build
2009 - 10 - 04 Geoff Norton < gnorton @ novell . com >
2008 - 10 - 12 Carlos Alberto Cortez < calberto . cortez @ gmail . com >
Add unit tests specific for RC2CryptoServiceProvider and TripleDESCryptoServiceProvider implementations ( i . e . not the algorithms )
Revert
Fix the build if - j4 is used .
2002 - 10 - 21 Jeffrey Stedfast < fejj @ ximian . com >
System . Xml . dll . sources : Added SerializationSource . cs .
[ sgen ] Remove CARDTABLE_STATS
Revert " Merge pull request from panzone / los_partial_marking "
[ aot ] Add bidi binding for Promise / Task .
* EventLog . cs : Fixed Designer attribute to match MS . NET .
2008 - 07 - 09 Zoltan Varga < vargaz @ gmail . com >
[ runtime ] Fix an unreachable code part in mini_emit_memset ( ) . Fixes . Thanks to Bill Seurer for spotting it .
2008 - 12 - 26 Zoltan Varga < vargaz @ gmail . com >
2005 - 04 - 24 Atsushi Enomoto < atsushi @ ximian . com >
removed test file
New tests .
[ sgen ] Concurrent flag instead of just char * .
2002 - 10 - 20 Ravi Pratap < ravi @ ximian . com >
2006 - 07 - 30 Carlos Alberto Cortez < calberto . cortez @ gmail . com >
[ threadpool - io ] Remove sparse - array for poll backend
[ netcore ] Fix the test . ( )
2004 - 09 - 12 Zoltan Varga < vargaz @ freemail . hu >
new test
* TypeDef . cs ( TypeDef . AddFieldDef ) : Throw exception if duplicate field being
[ jit ] Allocate thunk space for tail calls . Fixes .
Bump ikvm
2004 - 09 - 14 Miguel de Icaza < miguel @ ximian . com >
2005 - 04 - 06 Marek Safar < marek . safar @ seznam . cz >
Remove unused file
Merge pull request from monojenkins / sync - pr - 34600 - from - runtime
2007 - 10 - 19 Chris Toshok < toshok @ ximian . com >
2009 - 04 - 02 Marek Habersack < mhabersack @ novell . com >
2003 - 09 - 13 Gonzalo Paniagua Javier < gonzalo @ ximian . com >
Merge pull request from ntherning / fix - assert - in - poll_event_wait - on - WSAENOTSOCK - on - windows
[ netcore ] Fix a test . It was broken in so many ways , but hidden behind a check that disabled it everywhere . : (
2005 - 02 - 09 Chris Toshok < toshok @ ximian . com >
2006 - 08 - 19 Chris Toshok < toshok @ ximian . com >
2005 - 03 - 04 Gonzalo Paniagua Javier < gonzalo @ ximian . com >
2008 - 04 - 28 Zoltan Varga < vargaz @ gmail . com >
2006 - 02 - 13 Chris Toshok < toshok @ ximian . com >
2003 - 09 - 12 Martin Baulig < martin @ ximian . com >
[ runtime ] Register get gsharedvt arg trampolines
[ monkeydoc ] Avoid extra copying of EcmaDesc
2006 - 10 - 04 Miguel de Icaza < miguel @ novell . com >
Tiny : unreadable and additions
[ corlib ] Interning of utc literals in TimeZoneInfo conversion . Fixes
updates from grammar changes
* HttpWebClientProtocol . cs : Obey rules for fixed sized windows ( no
Initialization of default proxy added .
DIST_SUBDIRS is a thing , fixes parallelism broken by previous change
2005 - 07 - 07 Lluis Sanchez Gual < lluis @ novell . com >
[ sgen ] Add debug option for valloc limit
2008 - 01 - 02 Jb Evain < jbevain @ novell . com >
* FocusTest . cs : Added test for .
* nunit20 / util / Makefile ( EXTRA_DISTFILES ) : Remove
* main . c ( dis_method_list ) : Oops . Check method_context for null .
Fix Debug . Assert in GenericDelegateCache ( dotnet / coreclr )
Fixed comment .
Make redraw accessible and nicer
Merge pull request from kitsilanosoftware / rpm - spec - dead - entries
2007 - 01 - 13 Atsushi Enomoto < atsushi @ ximian . com >
[ Coop ] Convert System . Security . Cryptography . ( )
removed System . Net . Http . Headers
2008 - 01 - 06 Zoltan Varga < vargaz @ gmail . com >
Tue Jul 19 19 : 47 : 09 CEST 2002 Paolo Molaro < lupus @ ximian . com >
Ensure the condition
2008 - 05 - 14 Zoltan Varga < vargaz @ gmail . com >
Fixed some counters under 32bits .
2010 - 04 - 01 Zoltan Varga < vargaz @ gmail . com >
2008 - 04 - 02 Ivan N . Zlatev < contact @ i - nz . net >
2007 - 03 - 11 Marek Safar < marek . safar @ gmail . com >
2009 - 07 - 22 Sebastien Pouliot < sebastien @ ximian . com >
2008 - 05 - 02 Miguel de Icaza < miguel @ novell . com >
2009 - 10 - 16 Marek Safar < marek . safar @ gmail . com >
2005 - 12 - 06 Gonzalo Paniagua Javier < gonzalo @ ximian . com >
refine nits
2008 - 09 - 27 Zoltan Varga < vargaz @ gmail . com >
* MethodInstr . cs : callvirt operation is implicitly instance .
2002 - 10 - 31 Tim Coleman ( tim @ timcoleman . com )
* System . Data . Forms / GridEntry . cs : Add a null check .
2002 - 06 - 30 Gonzalo Paniagua Javier < gonzalo @ ximian . com >
Update the comments for mono_use_llvm .
2003 - 04 - 17 Gonzalo Paniagua Javier < gonzalo @ ximian . com >
[ corlib ] Match closer ExecutionContextSwitcher methods
2006 - 05 - 12 Peter Dennis Bartok < pbartok @ novell . com >
TARGET_J2EE : GetElement , 51422
2006 - 04 - 12 Zoltan Varga < vargaz @ gmail . com >
[ runtime ] Get rid of the create_delegate_trampoline_data ( ) helper function which only has one caller .
[ profiler ] Emit the seq point code used for soft breakpoints .
2005 - 11 - 07 Atsushi Enomoto < atsushi @ ximian . com >
2005 - 04 - 25 Marek Safar < marek . safar @ seznam . cz >
2004 - 12 - 19 Atsushi Enomoto < atsushi @ ximian . com >
svn path = / trunk / mcs / ; revision = 51213
New test .
Attempt to fix
2005 - 06 - 04 Dick Porter < dick @ ximian . com >
2002 - 07 - 24 Miguel de Icaza < miguel @ ximian . com >
Cache the Unreal profile
Merge pull request from vargaz / wasm - xunit - 4
Forgot to commit this
2004 - 09 - 28 Martin Baulig < martin @ ximian . com >
[ runtime ] Use', "'", 'this_obj', "'", 'instead of', "'", 'this', "'", 'in a few more places .
ongoing endpoint implementation .
2007 - 08 - 04 Miguel de Icaza < miguel @ novell . com >
[ arm ] Fix the build .
MemoryExtensions ToUpper / ToLower throws for overlapping buffer ( )
2008 - 07 - 08 Rodrigo Kumpera < rkumpera @ novell . com >
2003 - 07 - 21 Gaurav Vaish < gvaish_mono AT lycos . com >
2002 - 05 - 06 Nick Drochak < ndrochak @ gol . com >
2008 - 05 - 05 Zoltan Varga < vargaz @ gmail . com >
2008 - 01 - 09 Zoltan Varga < vargaz @ gmail . com >
2005 - 04 - 19 Martin Baulig < martin @ ximian . com >
[ System . XML ] Fix handling of global variables in XSLs ( )
[ wasm ] [ debugger ] Add support for surfacing inherited members ( )
[ runtime ] Remove use of mono_raise_exception in mono_object_isinst_mbyref_checked
2004 - 09 - 12 Gonzalo Paniagua Javier < gonzalo @ ximian . com >
New tests .
2006 - 10 - 03 Chris Toshok < toshok @ ximian . com >
2006 - 09 - 15 Igor Zelmanovich < igorz @ mainsoft . com >
2002 - 07 - 17 Martin Baulig < martin @ gnome . org >
2009 - 10 - 29 Jonathan Pobst < monkey @ jpobst . com >
* test - csproj2k5 - 050 . cs : Set output to xml - 050 . xml , not xml - 050 . cs .
[ jit ] Fix the EqualityComparer . get_Default ( ) optimization for now , it seems to cause failures in the corlib tests on the ios simulator . ( )
new test
2002 - 06 - 21 Martin Baulig < martin @ gnome . org >
[ sgen ] Heavy hash table statistics .
2007 - 03 - 30 Atsushi Enomoto < atsushi @ ximian . com >
[ Coop ] ves_icall_RuntimeFieldInfo_GetRawConstantValue . ( )
[ bcl ] Update BCL Linked Size
- Part of the previus checkin , missed these files
2006 - 12 - 08 Zoltan Varga < vargaz @ gmail . com >
2004 - 09 - 20 Sebastien Pouliot < sebastien @ ximian . com >
[ mcs ] Filter out nested private types of forwarded types
2005 - 05 - 04 Alexander Olk < xenomorph2 @ onlinehome . de >
* * * empty log message * * *
[ Tailcall ] Proper fix for F # regression . ( )
2003 - 09 - 21 Martin Baulig < martin @ ximian . com >
2008 - 04 - 06 Rodrigo Kumpera < rkumpera @ novell . com >
2002 - 02 - 23 Tim Coleman < tim @ timcoleman . com >
Merge pull request from esdrubal / servicemodel
2004 - 04 - 05 Sebastien Pouliot < sebastien @ ximian . com >
oops
2005 - 02 - 04 Jonathan Chambers < jonathan . chambers @ ansys . com >
2008 - 03 - 25 Sebastien Pouliot < sebastien @ ximian . com >
Added some files
[ sgen ] Precisely log the size of the promoted objects during minors
2006 - 03 - 09 Alexander Olk < alex . olk @ googlemail . com >
add MemberBind tests
2008 - 04 - 28 Jonathan Pobst < monkey @ jpobst . com >
[ sgen ] Concurrent flag instead of just char * .
* Mono . Mozilla / DOM / Node . cs , Mono . Mozilla / DOM / Window . cs : Fix comparison operators
* UnixEnvironment . cs ( ParseSolution ) : Report warning instead of error
2004 - 07 - 14 Gonzalo Paniagua Javier < gonzalo @ ximian . com >
ChangeLog : Updated ChangeLog .
2008 - 10 - 29 Sebastien Pouliot < sebastien @ ximian . com >
2006 - 02 - 04 Chris Toshok < toshok @ ximian . com >
2008 - 09 - 28 Jb Evain < jbevain @ novell . com >
2002 - 10 - 15 Nick Drochak < ndrochak @ gol . com >
Fix the build .
* FocusTest . cs : Ignore tests that rely on Form . Show ( ) synchronously
Fix race related deserialization
2008 - 10 - 09 Sebastien Pouliot < sebastien @ ximian . com >
2008 - 01 - 06 Ivan N . Zlatev < contact @ i - nz . net >
2007 - 06 - 25 Atsushi Enomoto < atsushi @ ximian . com >
* * * * Merged r43640 from MCS * * * *
[ coop ] Remove try { . . . } finally { . . . } from managed - to - native wrapper
Bump build snapshot submodule
2007 - 05 - 18 Jonathan Pobst < monkey @ jpobst . com >
2009 - 10 - 20 Zoltan Varga < vargaz @ gmail . com >
2005 - 01 - 15 Jordi Mas i Hernandez < jordi @ ximian . com >
[ wasm ] [ debugger ] Clear valuetypes cache on Debugger . resume ( )
2003 - 04 - 10 Gonzalo Paniagua Javier < gonzalo @ ximian . com >
* ThemeTest . cs : tests added
2005 - 09 - 13 Atsushi Enomoto < atsushi @ ximian . com >
2003 - 11 - 13 Martin Baulig < martin @ ximian . com >
[ wasm ] [ debugger ] Clear valuetypes cache on Debugger . resume ( )
2004 - 12 - 09 Atsushi Enomoto < atsushi @ ximian . com >
2005 - 11 - 24 Atsushi Enomoto < atsushi @ ximian . com >
2006 - 01 - 12 Dick Porter < dick @ ximian . com >
[ Tailcall ] ves_icall_RuntimeFieldInfo_GetRawConstantValue . ( )
2008 - 06 - 07 Atsushi Enomoto < atsushi @ ximian . com >
[ System . Runtime . CoreLib ] Avoid " invalid source table files with bulk inserts
[ wasm ] Fix the AOT cache check on wasm as well . ( )
[ System . XML ] Make test port from 8080 to 30158
Add mono .
2004 - 07 - 24 Martin Baulig < martin @ ximian . com >
2007 - 10 - 09 Marek Safar < marek . safar @ gmail . com >
[ jit ] Fix the loadable llvm build .
2002 - 04 - 22 Dick Porter < dick @ ximian . com >
2003 - 10 - 04 Gonzalo Paniagua Javier < gonzalo @ ximian . com >
2003 - 06 - 24 Gonzalo Paniagua Javier < gonzalo @ ximian . com >
Runtime with new page
refactoring
Nullable : Int32 ( dotnet / coreclr )
Merge pull request from akoeplinger / fix - ipv6 - teredo
[ System . EnterpriseServices ] Rename method parameters to match . NET contract
[ profiler ] Clean up usage of signature variables .
Add new logo
Move Interlocked
Remove gdiplus file
[ eglib ] Fix the embedding of i8 .
* NumberFormatter . cs : Fixed rounding for float and the string output now includes all the precision ( not counting preceding zeros ) . This fix the DecimalTest . TestConstructSingleRounding_NotWorking test cases .
2008 - 07 - 02 Carlos Alberto Cortez < calberto . cortez @ gmail . com >
[ sgen ] Heavy hash table statistics .
2005 - 07 - 09 Chris Toshok < toshok @ ximian . com >
[ runtime ] Use mono_set_pending_exception ( ) in some places in icalls .
Merge pull request from esdrubal / utf8encoding
2005 - 09 - 14 Gonzalo Paniagua Javier < gonzalo @ ximian . com >
Merge pull request from mattleibow / master
* appdomain . c : Use mono_get_exception_bad_image_format2 to construct
2003 - 11 - 25 Zoltan Varga < vargaz @ freemail . hu >
2009 - 04 - 25 Geoff Norton < gnorton @ novell . com >
[ mcs ] eval . cs : add ValueModificationHandler delegate to be more clear
[ wasm ] Adjust . SpinCount is a lot of NULL
Revert " [ marshal ] delay dummy string allocation "
[ reflection ] Skip the copying of EcmaDesc
[ sdks ] Add missing legacy GetTitle method
Wed Jun 3 12 : 57 : 31 CEST 2002 Paolo Molaro < lupus @ ximian . com >
* ILParser . jay : Fix casting for data type constants .
2006 - 11 - 22 Gonzalo Paniagua Javier < gonzalo @ ximian . com >
2009 - 03 - 01 Zoltan Varga < vargaz @ gmail . com >
2006 - 05 - 25 Alexander Olk < alex . olk @ googlemail . com >
[ runtime ] Added - - with - cross - offsets configure option for including an explicit AOT cross offsets file .
* DataGrid . cs : Minor fixes .
2002 - 09 - 11 Martin Baulig < martin @ gnome . org >
[ System . Private . CoreLib ] Add Nullable runtime support
2010 - 05 - 05 Marek Habersack < mhabersack @ novell . com >
2010 - 01 - 30 Rodrigo Kumpera < rkumpera @ novell . com >
2002 - 10 - 30 Gonzalo Paniagua Javier < gonzalo @ ximian . com >
[ profiler ] Add myself as author .
2007 - 04 - 31 Andreas Faerber < andreas . faerber @ web . de >
add NewArrayInit tests
Changes sourceStream . Read correctly .
Populate properties to match MS . NET
* * * empty log message * * *
Small fix .
Bump linker
[ ci ] Turn off Boehm on Windows x64 gcc build
Mon Oct 6 17 : 59 : 53 CEST 2002 Paolo Molaro < lupus @ ximian . com >
Merge pull request from Unroll - Koenig / master
2006 - 04 - 09 Miguel de Icaza < miguel @ novell . com >
2004 - 04 - 30 Gonzalo Paniagua Javier < gonzalo @ ximian . com >
2003 - 04 - 09 Zoltan Varga < vargaz @ freemail . hu >
2008 - 04 - 25 Rodrigo Kumpera < rkumpera @ novell . com >
Fix System . ComponentModel . DataErrorsChangedEventArgs API .
2008 - 10 - 31 Atsushi Enomoto < atsushi @ ximian . com >
* checks . cs ( MemberAccess . ResolveMemberAccess ) :
2009 - 10 - 08 Atsushi Enomoto < atsushi @ ximian . com >
2010 - 03 - 09 Gonzalo Paniagua Javier < gonzalo @ novell . com >
2006 - 02 - 10 Peter Dennis Bartok < pbartok @ novell . com >
2007 - 12 - 29 Sebastien Pouliot < sebastien @ ximian . com >
2004 - 12 - 01 Atsushi Enomoto < atsushi @ ximian . com >
2001 - 11 - 28 Miguel de Icaza < miguel @ ximian . com >
[ utils ] Add missing legacy GetTitle method
[ sgen ] Heavy statistics counters for scanned object descriptors .
2007 - 09 - 10 Atsushi Enomoto < atsushi @ ximian . com >
Fix typo in changelog
Bump ikvm
2007 - 01 - 31 Marek Sieradzki < marek . sieradzki @ gmail . com >
Fix build .
[ jit ] Fix the EqualityComparer . get_Default ( ) optimization for now , it seems to cause failures in the corlib tests on the ios simulator . ( )
2008 - 09 - 25 Jb Evain < jbevain @ novell . com >
2009 - 04 - 12 Miguel de Icaza < miguel @ novell . com >
Tiny fix :
2008 - 03 - 17 Zoltan Varga < vargaz @ gmail . com >
[ corlib ] Fix culture info error message . Fixes
2008 - 07 - 11 Atsushi Enomoto < atsushi @ ximian . com >
2004 - 07 - 21 Zoltan Varga < vargaz @ freemail . hu >
[ Mono . Security ] : Add API to let MonoTlsProvider provider a custom system certificate validator .
2004 - 09 - 12 Marek Safar < marek . safar @ seznam . cz >
2010 - 05 - 10 Rodrigo Kumpera < rkumpera @ novell . com >
* CompareInfo . cs ( operator = = , operator ! = ) : Implement .
A message update
Add DecimalFormatterTest . cs from Patrick Kalkman
NO ARGS : System / ArrayTest . cs : ( 36 )
* Initial checkin for a number of enums *
2002 - 03 - 27 Martin Baulig < martin @ gnome . org >
2009 - 04 - 30 Jb Evain < jbevain @ novell . com >
2005 - 01 - 31 Zoltan Varga < vargaz @ freemail . hu >
2002 - 10 - 17 Martin Baulig < martin @ gnome . org >
Add new counter type method
Mon Dec 6 16 : 43 : 37 CET 2003 Paolo Molaro < lupus @ ximian . com >
[ corlib ] Fix culture info error message . Fixes
[ runtime ] Fail classes that have a field with a bad underlying type
* HttpWorkerRequest . cs : Removed TODO for 2 . 0 APIs and return the default values for them .
2008 - 05 - 25 Ivan N . Zlatev < contact @ i - nz . net >
fix credits
q
* TreeView . cs : Added
adjust
Fix
2008 - 02 - 21 Rodrigo Kumpera < rkumpera @ novell . com >
2004 - 08 - 05 Zoltan Varga < vargaz @ freemail . hu >
Merge pull request from monojenkins / update - csprojs
* ComboBox . cs : Forward ContextMenu to the underlying textbox , if any .
[ w32handle ] Fill in the presence of canaries .
svn path = / trunk / mcs / ; revision = 53056
Improve test .
2007 - 02 - 02 Sebastien Pouliot < sebastien @ ximian . com >
Wed Sep 3 17 : 47 : 15 CEST 2005 Paolo Molaro < lupus @ ximian . com >
2008 - 06 - 24 Zoltan Varga < vargaz @ gmail . com >
Merge pull request from esdrubal / xmlgenerator
2008 - 08 - 15 Zoltan Varga < vargaz @ gmail . com >
[ Debugger ] Remove use of w32handle for File , Console , Pipe and Socket ( )
Add new logo
Include the spec . c .
Flush
* FieldDef . cs : Set the underlying type for bytearray data constants . This
2009 - 08 - 09 Jb Evain < jbevain @ novell . com >
2006 - 02 - 28 Martin Baulig < martin @ ximian . com >
2009 - 07 - 21 Zoltan Varga < vargaz @ gmail . com >
Mon Aug 14 16 : 47 : 39 CEST 2003 Paolo Molaro < lupus @ ximian . com >
2010 - 06 - 15 Marek Habersack < mhabersack @ novell . com >
2002 - 04 - 02 Miguel de Icaza < miguel @ ximian . com >
2010 - 01 - 28 Rodrigo Kumpera < rkumpera @ novell . com >
[ wasm ] [ debugger ] Clear valuetypes cache on Debugger . resume ( )
2009 - 05 - 06 Atsushi Enomoto < atsushi @ ximian . com >
2010 - 03 - 06 Zoltan Varga < vargaz @ gmail . com >
Fix bug .
2008 - 10 - 14 Marek Habersack < mhabersack @ novell . com >
Wed Apr 4 18 : 35 : 25 CEST 2002 Paolo Molaro < lupus @ ximian . com >
2004 - 05 - 15 Atsushi Enomoto < atsushi @ ximian . com >
2008 - 06 - 11 Zoltan Varga < vargaz @ gmail . com >
Merge pull request from akoeplinger / fix - processtest - race
Merge pull request from monojenkins / bclsizediff
* ComboBoxTest . cs : Added test for bug .
undo .
[ sgen ] Fix race condition in mono_gc_weak_link_get ( ) .
cleanup
2007 - 10 - 28 Marek Safar < marek . safar @ gmail . com >
2010 - 06 - 10 Marek Safar < marek . safar at gmail . com >
2006 - 11 - 22 Mike Kestner < mkestner @ novell . com >
2002 - 03 - 26 Gaurav Vaish < gvaish @ iitk . ac . in >
[ runtime ] Remove unused dead code .
[ reflection ] Harder the custom attribute code against common type load errors .
[ profiler ] Clarify the root_type size code to uleb128 .
2005 - 10 - 09 Gonzalo Paniagua Javier < gonzalo @ ximian . com >
* HttpRequest . cs : Cleanup method .
Visual Studio build support : updated the scripts that generate csproj files .
2002 - 07 - 12 Gonzalo Paniagua Javier < gonzalo @ ximian . com >
2009 - 01 - 28 Veerapuram Varadhan < vvaradhan @ novell . com >
- Added Locale for ColorDialog ( de - DE )
typo fix
2004 - 09 - 15 Atsushi Enomoto < atsushi @ ximian . com >
Replaced the System . Core build for the monotouch profile
* * * empty log message * * *
[ Tailcall ] Proper fix for F # regression . ( )
[ bcl ] Fix Console support on ios , the TextWriter class from the reference sources called TextWriter : Write ( char ) , so stdout was never flushed .
2004 - 01 - 25 Zoltan Varga < vargaz @ freemail . hu >
2003 - 04 - 21 Piers Haken < piersh @ friskit . com >
2008 - 08 - 30 Rodrigo Kumpera < rkumpera @ novell . com >
[ runtime ] Move pending_exception from InternalThread to Thread to avoid crashing on domain unload .
2006 - 10 - 02 Zoltan Varga < vargaz @ gmail . com >
2004 - 01 - 24 Tim Coleman < tim @ timcoleman . com >
Revert this part the previous commit .
[ runtime ] Work around an ios9 problem where dlopen ( ) cant', "'", 'find system libraries specified without paths .
2004 - 05 - 14 Atsushi Enomoto < atsushi @ ximian . com >
[ Cleanup ] Remove windows - specific shutdown hackaround .
[ test ] Minor fixes around System . Reflection ( )
2003 - 04 - 11 Piers Haken < piersh @ friskit . com >
* * * empty log message * * *
[ SRE ] Remove unused icall from inside a DISABLE_REFLECTION_EMIT block .
2002 - 10 - 9 Daniel Morgan < danmorg @ sc . rr . com >
[ runtime ] Add new tests for the g_shell_parse
[ runtime ] Remove use of mono_raise_exception in mono_object_isinst_mbyref_checked
2008 - 03 - 02 Ivan N . Zlatev < contact @ i - nz . net >
2003 - 05 - 12 Zoltan Varga < vargaz @ freemail . hu >
Merge pull request from robwilkens / master
Fix : Removed wrong increment
Revert
[ mcs ] eval . cs : add ValueModificationHandler delegate to be more clear
2002 - 03 - 11 Jason Diamond < jason @ injektilo . org >
* Credited . cs , HtmlElement . cs , SoapMethodAttribute . cs ,
2010 - 02 - 25 Atsushi Enomoto < atsushi @ ximian . com >
2009 - 11 - 04 Rodrigo Kumpera < rkumpera @ novell . com >
2004 - 12 - 09 Atsushi Enomoto < atsushi @ ximian . com >
2008 - 08 - 04 Rodrigo Kumpera < rkumpera @ novell . com >
Oops
2009 - 05 - 06 Gonzalo Paniagua Javier < gonzalo @ novell . com >
2004 - 09 - 11 Zoltan Varga < vargaz @ freemail . hu >
[ Tailcall ] ves_icall_RuntimeFieldInfo_GetRawConstantValue . ( )
2008 - 03 - 06 Andreas Nahr < ClassDevelopment @ A - SoftTech . com >
* InitializationEventAttribute . cs : New .
2002 - 09 - 16 Dietmar Maurer < dietmar @ ximian . com >
Merge pull request from ntherning / enable - building - bcl - with - mono - 4 . 6 - mcs
[ jit ] Fix the check for calling mono_get_vtable_var ( ) , check for', "'", 'cfg - > method', "'", 'instead of', "'", 'method', "'", ', the latter is the inlined method . ( )
2004 - 12 - 02 Sebastien Pouliot < sebastien @ ximian . com >
2002 - 04 - 14 Radek Doulik < rodo @ ximian . com >
svn path = / trunk / mcs / ; revision = 74905
2002 - 04 - 27 Miguel de Icaza < miguel @ ximian . com >
Fix badge usage in install_eh_callback . ( )
A test for bug
* * * * Merged r41636 from MCS * * * *
2009 - 01 - 22 Carlos Alberto Cortez < calberto . cortez @ gmail . com >
Merge pull request from Therzok / addSealed
2005 - 06 - 27 Gonzalo Paniagua Javier < gonzalo @ ximian . com >
[ io - layer ] Make _WAPI_PRIVATE_MAX_SLOTS and _WAPI_HANDLE_INITIAL_COUNT definition private
2007 - 10 - 01 Sebastien Pouliot < sebastien @ ximian . com >
2010 - 03 - 01 Geoff Norton < gnorton @ novell . com >
2008 - 03 - 04 Atsushi Enomoto < atsushi @ ximian . com >
2009 - 04 - 15 Carlos Alberto Cortez < calberto . cortez @ gmail . com >
[ runtime ] Make the code in mini - llvm . c more C + + compatible .
2006 - 09 - 19 Marek Sieradzki < marek . sieradzki @ gmail . com >
2005 - 04 - 10 Sebastien Pouliot < sebastien @ ximian . com >
2006 - 07 - 01 Geoff Norton < gnorton @ customerdna . com >
2008 - 01 - 28 Marek Habersack < mhabersack @ novell . com >
2004 - 04 - 21 Geoff Norton < gnorton @ customerdna . com >
build error here , revert until fixed
* SqlParameter . cs ( ConvertToFrameworkType ) : Only wrap FormatException
2001 - 12 - 15 Miguel de Icaza < miguel @ ximian . com >
* dis - cil . c ( get_encoded_user_string ) : use g_strescape instead of
2009 - 01 - 17 Rodrigo Kumpera < rkumpera @ novell . com >
[ llvm ] Fix the loadable llvm build .
2007 - 05 - 14 Martin Baulig < martin @ ximian . com >
* SortedDictionary . cs ( ICollection . Contains ) : Use EqualityComparer
[ ci ] Add assertion for Math operations .
2009 - 01 - 24 Zoltan Varga < vargaz @ gmail . com >
fix filename
2002 - 12 - 17 Martin Baulig < martin @ gnome . org >
2007 - 10 - 28 Martin Baulig < martin @ ximian . com >
2007 - 25 - 17 Chris Toshok < toshok @ ximian . com >
[ System . XML ] Fix handling of global variables in XSLs ( )
2010 - 04 - 29 Zoltan Varga < vargaz @ gmail . com >
New tests .
System . Data . dll . sources : Added DbException . cs , SqlClientMetaDataCollectionNames . cs
Add aspnetwebstack submodule
2006 - 07 - 05 Chris Toshok < toshok @ ximian . com >
fix tests :
Tweak the build on - .
Add new files
* TabControl . cs : Stay in bounds when cycling . Make sure Handled is
2003 - 02 - 09 Sebastien Pouliot < spouliot @ videotron . ca >
Do not reject contract from the same method , could happen by a type with multiple service contracts within the typie hierarchy .
2005 - 02 - 10 Marek Safar < marek . safar @ seznam . cz >
2007 - 10 - 04 Miguel de Icaza < miguel @ novell . com >
[ runtime ] Fix a warning .
Moved deprecated library
[ sdks ] Fix the build .
This is not supported now
2010 - 05 - 12 Marek Safar < marek . safar @ gmail . com >
2008 - 05 - 28 Jonathan Pobst < monkey @ jpobst . com >
Fix a warning .
2005 - 09 - 06 Marek Safar < marek . safar @ seznam . cz >
* System . dll . sources : added CompilationSection . cs
Fri Aug 4 16 : 12 : 43 CEST 2007 Paolo Molaro < lupus @ ximian . com >
Oops
Fix', ['Oops
2002 - 05 - 14 Nick Drochak < ndrochak @ gol . com >
Invert the condition
[ sgen ] Fix invalid codegen for FastAllocateString .
2005 - 03 - 29 Atsushi Enomoto < atsushi @ ximian . com >
Merge pull request from alexrp / master
Revert r84984 and r85314
Fix reflection build under 36b9052157da87809210eb3cb00b9c5a8af03823 .
2005 - 06 - 10 Peter Bartok < pbartok @ novell . com >
2005 - 05 - 10 Peter Bartok < pbartok @ novell . com >
switch to mono implementation for SqlTypes
Merge pull request from akoeplinger / networkcredential - securestring
[ arm ] Fix crashes on ios9 by using 16kb trampoline pages .
2005 - 06 - 04 Duncan Mak < duncan @ novell . com >
2002 - 04 - 16 Dietmar Maurer < dietmar @ ximian . com >
2006 - 09 - 23 Martin Baulig < martin @ ximian . com >
2002 - 03 - 15 Gonzalo Paniagua Javier < gonzalo @ ximian . com >
Cleanup WindowsBase test failures .
2003 - 09 - 10 Andreas Nahr < ClassDevelopment @ A - SoftTech . com >
Update
[ corlib ] Restores ConcurrentDictionary serialization support .
Added new missing attributes ( dotnet / coreclr )
[ sgen ] GrayQueueEntry struct instead of just char * .
* main . c : Spaces after visibility types .
2008 - 09 - 04 George Giolfan < georgegiolfan @ yahoo . com >
[ 642437 ] Nested types have to check type arguments only and not full class
Merge pull request from HorstKakuschke / master
* TabControl . cs : Use dashed lines to connect nodes . Use the
2002 - 09 - 20 Atsushi Enomoto < ginga @ kit . hi - ho . ne . jp >
2007 - 09 - 28 Atsushi Enomoto < atsushi @ ximian . com >
Up
build error here , revert until fixed
[ corlib ] Match closer ExecutionContextSwitcher methods
* RemotingConfiguration . cs : Use GetCachedSoapAttribute ( ) .
[ mono ] Implement Arm intrinsics : ArmBase , Crc32 ( )
[ runtime ] Add System . Cecil to Mono . CSharp . dll .
tramp - arm64 - gsharedvt . c : In function', "'", 'mono_arch_get_gsharedvt_trampoline', "'", ':
2008 - 01 - 26 Jonathan Pobst < monkey @ jpobst . com >
2007 - 01 - 16 Igor Zelmanovich < igorz @ mainsoft . com >
2004 - 09 - 04 Atsushi Enomoto < atsushi @ ximian . com >
* MenuAdapterTest . cs ( MyMenuAdapter ) : Add new forwarding methods
Use lower flags for unmanaged methods .
Fix gtest - 284 failure in net_4_0 profile
Merge pull request from Therzok / addSealed
2002 - 12 - 04 Daniel Morgan < danmorg @ sc . rr . com >
2008 - 05 - 19 George Giolfan < georgegiolfan @ yahoo . com >
[ genmdesc ] Improved instruction length error handling for C version .
Modified extensions to VARCHAR type
Revert accidental r53054
[ new ] Add an explicit null check to one of the unsafe string cctors instead of relaying on the implicit null check in memcpy ( ) to easy debugging .
2005 - 02 - 17 Atsushi Enomoto < atsushi @ ximian . com >
[ tests ] Xamarin . Android has the desktop behavior
2002 - 05 - 11 Martin Baulig < martin @ gnome . org >
[ Mono . Security . Providers . OldTls ] : Don', "'", 't build on mobile .
Fix
Prepare the System . Core build for the monotouch profile
2009 - 02 - 01 Atsushi Enomoto < atsushi @ ximian . com >
2005 - 04 - 01 Geoff Norton < gnorton @ customerdna . com >
2004 - 05 - 05 Zoltan Varga < vargaz @ freemail . hu >
Merge pull request from radical / asm_remap
[ mini ] Implement Arm intrinsics : ArmBase , Crc32 ( )
[ System ] Add IPv6 support to FtpWebRequest
2005 - 12 - 11 Gonzalo Paniagua Javier < gonzalo @ ximian . com >
2006 - 04 - 07 Zoltan Varga < vargaz @ gmail . com >
2010 - 06 - 09 Mark Probst < mark . probst @ gmail . com >
basic summaries
2003 - 01 - 15 Ville Palo < vi64pa @ koti . soon . fi >
* Driver . cs : Fixed AssemblyDescription to match MS . NET .
2005 - 10 - 25 Ben Maurer < bmaurer @ ximian . com >
2007 - 06 - 14 Jb Evain < jbevain @ novell . com >
Disable some deadlocking plinq tests
[ llvm ] Fix the loadable llvm build .
Test to reveal defect in XmlSerializer .
[ corlib ] Interning of utc literals in TimeZoneInfo conversion . Fixes
[ Coop ] ves_icall_RuntimeFieldInfo_GetRawConstantValue . ( )
2004 - 06 - 09 Zoltan Varga < vargaz @ freemail . hu >
2002 - 07 - 17 Martin Baulig < martin @ gnome . org >
[ bcl ] Fix a test . It was broken in so many ways , but hidden behind a check that disabled it everywhere . : (
1 . 0 . 1
[ threadpool - io ] Remove sparse - array for poll backend
install
- Duh !
2005 - 05 - 22 Zoltan Varga < vargaz @ freemail . hu >
2006 - 05 - 12 Alexander Olk < alex . olk @ googlemail . com >
2010 - 02 - 11 Sebastien Pouliot < sebastien @ ximian . com >
Memory Implementation
Fix a warning if MONO_FAST_TLS_DECLARE is not defined .
[ ci ] Turn off Boehm on Windows x64 gcc build
2004 - 04 - 13 Zoltan Varga < vargaz @ freemail . hu >
Reverted those
2009 - 10 - 29 Marek Habersack < mhabersack @ novell . com >
[ System . Net . Http ] Incomplete byte ranges parsing . Fixes
9 - 03 - 09 Miguel de Icaza < miguel @ ximian . com >
typo fix
2007 - 07 - 15 Jonathan Chambers < joncham @ gmail . com >
[ netcore ] Generate overwriting EMSCRIPTEN_SDK_DIR . Add a runtime build with threads support if ENABLE_WASM_THREADS is defined in Make . config .
2006 - 01 - 06 Atsushi Enomoto < atsushi @ ximian . com >
* * * empty log message * * *
Moved the new images
Backport
refactoring
2005 - 03 - 04 Sebastien Pouliot < sebastien @ ximian . com >
new test
This one the build
* MoonIsolatedStorageFile . cs : Allow paths in search patterns .
Tue Jun 6 19 : 12 : 21 CEST 2002 Paolo Molaro < lupus @ ximian . com >
[ Mono . Debugger . Soft ] Catching AssemblyMirror in AppDomainMirror and TypeMirror in AssemblyMirror
Bump ikvm
2008 - 04 - 25 George Giolfan < georgegiolfan @ yahoo . com >
2007 - 11 - 01 Zoltan Varga < vargaz @ gmail . com >
[ genproj ] Move ToolsVersion 12
[ runtime ] Add support for using COM types from corlib .
2007 - 09 - 11 Zoltan Varga < vargaz @ gmail . com >
2009 - 07 - 31 Zoltan Varga < vargaz @ gmail . com >
2009 - 04 - 26 Marek Safar < marek . safar @ gmail . com >
Implemented NullTarget ( )
* ComboBoxTest . cs : Added DataBindingTest .
2006 - 01 - 09 Atsushi Enomoto < atsushi @ ximian . com >
Fix mobile build
Add stub for NET_2_0 LockBits
[ runtime ] Remove unused LLVM_MONO_BRANCH define .
2007 - 05 - 10 Jb Evain < jbevain @ novell . com >
2005 - 01 - 27 Cesar Lopez Nataren < cesar @ ciencias . unam . mx >
Checks the constructor declaring type .
[ corlib ] Fix Type : : GetEnumName / IsEnumDefined to return same value as Enum : : GetName / IsDefined
2006 - 04 - 02 Gonzalo Paniagua Javier < gonzalo @ ximian . com >
* RemotingServices . cs : Null values are now represented with DBNull instances .
* ILParser . jay : Create external method references
* Page . cs : Calculate the maximum and change vars
2003 - 11 - 04 Ville Palo < vi64pa @ kolumbus . fi >
Revert r46073 . cs
typo fix
[ sgen ] Remove CARDTABLE_STATS
2008 - 09 - 05 Sebastien Pouliot < sebastien @ ximian . com >
2004 - 06 - 01 Atsushi Enomoto < atsushi @ ximian . com >
fix typo
2004 - 12 - 13 Atsushi Enomoto < atsushi @ ximian . com >
Bump linker
2002 - 07 - 20 Dietmar Maurer < dietmar @ ximian . com >
2005 - 01 - 05 Sureshkumar T < tsureshkumar @ novell . com >
2005 - 11 - 14 Gonzalo Paniagua Javier < gonzalo @ ximian . com >
2002 - 07 - 11 Dietmar Maurer < dietmar @ ximian . com >
New tests .
* SqlDataReader . cs : Null values are now represented with DBNull instances .
2006 - 10 - 04 Miguel de Icaza < miguel @ novell . com >
ignore FSW test list
2001 - 07 - 15 Miguel de Icaza < miguel @ ximian . com >
2008 - 03 - 28 Zoltan Varga < vargaz @ gmail . com >
Bump bockbuild
2010 - 01 - 21 Miguel de Icaza < miguel @ novell . com >
2008 - 01 - 17 Marek Safar < marek . safar @ gmail . com >
[ xBuild ] Steps towards making pre - build scripts cross platform ; Lots more build now
2007 - 05 - 11 Marek Sieradzki < marek . sieradzki @ gmail . com > \\
2003 - 06 - 14 Patrik Torstensson
2009 - 04 - 21 Carlos Alberto Cortez < calberto . cortez @ gmail . com >
Merge pull request from lambdageek / dev / monoerror - stringy
2005 - 08 - 28 Sebastien Pouliot < sebastien @ ximian . com >
[ corlib ] Mobile build fixes
2006 - 10 - 02 Chris Toshok < toshok @ ximian . com >
2005 - 03 - 15 Atsushi Enomoto < atsushi @ ximian . com >
[ Tailcall ] Fill in some missing pieces : x86reg , armreg , armmembase , align x86patch , x86aot , armhardfloat . ( )
2006 - 04 - 05 Atsushi Enomoto < atsushi @ ximian . com >
Merge pull request from lambdageek / dev / monoerror - stringy
2005 - 05 - 05 Sebastien Pouliot < sebastien @ ximian . com >
2009 - 04 - 06 Atsushi Enomoto < atsushi @ ximian . com >
2003 - 05 - 01 Miguel de Icaza < miguel @ ximian . com >
[ WCF ] Update the tool name
Revert 73804
2009 - 07 - 03 Atsushi Enomoto < atsushi @ ximian . com >
[ mcs ] Reset anonymymous method with correct type arguments when async this requires hoisting in parent type . Fixes
2004 - 05 - 08 Umadevi S < sumadevi @ novell . com >
Flush
Patch .
2005 - 02 - 04 Zoltan Varga < vargaz @ freemail . hu >
* TemplateField . cs : fixed ExtractValuesFromCell , . net aspx parser not always creates ExtractTemplateValuesMethod delegate
2009 - 08 - 04 Rodrigo Kumpera < rkumpera @ novell . com >
* * * empty log message * * *
2002 - 07 - 12 Gaurav Vaish < gvaish_mono @ lycos . com >
2007 - 09 - 13 Rodrigo Kumpera < rkumpera @ novell . com >
2004 - 11 - 15 Atsushi Enomoto < atsushi @ ximian . com >
2002 - 09 - 02 Sebastien Pouliot < spouliot @ videotron . ca >
[ acceptance - tests ] Integrate Roslyn , CoreCLR and ms - test - suite into acceptance - tests
* System . Web . Services . dll . sources : Added new files :
2005 - 09 - 19 Chris Toshok < toshok @ ximian . com >
2008 - 02 - 14 Marek Habersack < mhabersack @ novell . com >
* PageCompiler . cs : Add Trace and TraceMode to framework initialize
2008 - 01 - 31 Stephane Delcroix < sdelcroix @ novell . com >
Disable the build .
DIST_SUBDIRS is a thing , fixes parallelism broken by previous change
Mono friendly MethodBase
Add new error
from head
svn path = / trunk / mcs / ; revision = 74905
this pass those
2004 - 01 - 26 Atsushi Enomoto < atsushi @ ximian . com >
2003 - 10 - 12 Igor Zelmanovich < igorz @ mainsoft . com >
2008 - 07 - 28 Marek Habersack < mhabersack @ novell . com >
remove redundand NotWorkings
2010 - 04 - 07 Atsushi Enomoto < atsushi @ ximian . com >
[ wasm ] Small packager to better work with our sample . Update the sample to use the binding engine and , finally , update the README with this all .
2003 - 09 - 02 Bernie Solomon < bernard @ ugsolutions . com >
2009 - 01 - 28 Atsushi Enomoto < atsushi @ ximian . com >
Fix the build .
2007 - 07 - 02 Marek Habersack < mhabersack @ novell . com >
[ netcore ] Update libtiff ( )
[ sgen ] GrayQueueEntry struct instead of just char * .
2007 - 10 - 15 Jonathan Chambers < joncham @ gmail . com >
2007 - 02 - 11 Igor Zelmanovich < igorz @ mainsoft . com >
2002 - 06 - 14 Gonzalo Paniagua Javier < gonzalo @ ximian . com >
2007 - 01 - 26 Igor Zelmanovich < igorz @ mainsoft . com >
2006 - 04 - 24 Alexander Olk < alex . olk @ googlemail . com >
2002 - 03 - 19 Gaurav Vaish < gvaish @ iitk . ac . in >
Tue Jun 15 16 : 36 : 51 CEST 2004 Paolo Molaro < lupus @ ximian . com >
2007 - 12 - 21 Nagappan A < anagappan @ novell . com >
[ runtime ] Use mono_debug_method_lookup_location ( ) instead of mono_debug_symfile_lookup_location ( ) , the latter doesn', "'", 't support portable pdb files . Use mono_debug_free_source_location ( ) for freeing . Fixes . ( )
PR feedback : GetElement ( 0 ) - > Sse2 . StoreLow
New test .
add Grasshopper - 41845 , 51422
Fix commit c31a58b6565c3d6a3994cac8bfd9e9eac1a54c71 .
In mcs :
[ wasm ] Revert CoreXml . ( )
2005 - 01 - 09 Sebastien Pouliot < sebastien @ ximian . com >
A message update
[ sgen ] Fix build on C99 .
[ csproj ] Update project files
* LateBinding . cs : Only stop server if it was actually created .
[ System . ServiceModel ] Changed specific xunit code
2005 - 09 - 04 Robert Jordan < robertj @ gmx . net >
2006 - 03 - 27 Rolf Bjarne Kvinge < RKvinge @ novell . com >
[ runtime ] Use MonoError for mono_class_vtable_full and mono_runtime_class_init_full .
2009 - 06 - 29 Marek Safar < marek . safar @ gmail . com >
[ System ] : Cleanup big WebConnectionStream cleanup . ( )
Dear God , make me recompile my code after removing debug * before * I check in
[ csproj ] Update project files
[ tests ] Xamarin . Android has the desktop behavior
2005 - 04 - 26 Martin Baulig < martin @ ximian . com >
2004 - 03 - 31 zovarga < vargaz @ freemail . hu >
2004 - 08 - 10 Atsushi Enomoto < atsushi @ ximian . com >
svn path = / trunk / mcs / ; revision = 74905
2003 - 04 - 06 Nick Drochak < ndrochak @ gol . com >
Mon May 16 15 : 01 : 34 CEST 2002 Paolo Molaro < lupus @ ximian . com >
[ jit ] Remove some dead code / includes .
[ wasm ] [ debugger ] Add support for surfacing inherited members ( )
2009 - 02 - 10 Jonathan Pobst < monkey @ jpobst . com >
Add new test
2009 - 01 - 12 Rodrigo Kumpera < rkumpera @ novell . com >
[ coop ] Check async suspend status in ReqSuspendInitSuspendBlocking case ( )
2002 - 11 - 02 Ravi Pratap < ravi @ ximian . com >
put in the right namespace
2008 - 12 - 03 Zoltan Varga < vargaz @ gmail . com >
2009 - 04 - 29 Gonzalo Paniagua Javier < gonzalo @ novell . com >
[ wasm ] [ debugger ] Add support for surfacing inherited members ( )
2009 - 11 - 04 Zoltan Varga < vargaz @ gmail . com >
[ tests ] Xamarin segfaults on Android , probably due to AppDomain . Unload
2003 - 03 - 20 Gonzalo Paniagua Javier < gonzalo @ ximian . com >
new API snapshot submodule
2008 - 10 - 21 George Giolfan < georgegiolfan @ yahoo . com >
2005 - 12 - 24 Atsushi Enomoto < atsushi @ ximian . com >
[ wasm ] [ debugger ] Hash messages and sessions directly ( )
[ interp ] Track
cls happyfication
Merge pull request from akoeplinger / fix - processtest - race
In System . Data . SqlClient :
Remove the Makefile
Tue Nov 12 11 : 56 : 42 CET 2002 Paolo Molaro < lupus @ ximian . com >
[ arm ] Allocate thunk space for tail calls . Fixes .
Small fix .
[ mini ] Remove CARDTABLE_STATS
Bump bockbuild
2007 - 02 - 16 Igor Zelmanovich < igorz @ mainsoft . com >
[ sgen ] Remove some dead code and data .
[ wasm ] [ debugger ] Clear valuetypes cache on Debugger . resume ( )
Oups , resgen was missing in 2 . 0
This is not supported now
2008 - 04 - 14 Jonathan Pobst < monkey @ jpobst . com >
Fix the build .
[ mini ] Add missing membars to the g_shell_parse
fix HintPath
2005 - 04 - 22 Atsushi Enomoto < atsushi @ ximian . com >
Merge pull request from kumpera / disable - owner
[ jit ] Rename MonoGenericParam - > serial to', "'", 'gshared_constraint', "'", ', use it to calculate object sizes / layouts correctly if a generic param matches a non - reference type during gsharing .
2005 - 08 - 22 Sureshkumar T < tsureshkumar @ novell . com >
2005 - 02 - 28 Lluis Sanchez Gual < lluis @ novell . com >
2008 - 04 - 19 Rodrigo Kumpera < rkumpera @ novell . com >
[ ci ] Turn off Boehm on Windows x64 gcc build
[ runtime ] Added gen - compact - seq - points option to MONO_DEBUG .
Merge pull request from ntherning / fix - assert - in - poll_event_wait - on - WSAENOTSOCK - on - windows
[ monkeydoc ] Add eglib support for bxc .
Add System . Data . Http . dll to the monotouch build .
[ jit ] Simplify the seq point code used for soft breakpoints .
2005 - 11 - 07 Daniel Morgan < danielmorgan @ verizon . net >
* MemberAttributes . cs : Added comment that " fixes " .
Add new test
By : Yonik < yonik @ mainsoft . com >
2005 - 04 - 19 Sebastien Pouliot < sebastien @ ximian . com >
missed one F
Dear God , make me recompile my code after removing debug * before * I check in
2001 - 09 - 22 Miguel de Icaza < miguel @ ximian . com >
[ aot ] Remove pre WinXP SP2 windows support code .
2004 - 09 - 29 Umadevi S < sumadevi @ novell . com >
2004 - 09 - 07 Sebastien Pouliot < sebastien @ ximian . com >
[ master ] Update dependencies files
Merge pull request from NoesisLabs / master
[ mkbundle ] Don', "'", 't warn about duplicate token for MonoArrayMethod
solution class initialization .
2009 - 10 - 29 Marek Habersack < mhabersack @ novell . com >
install
Fix the build .
2004 - 11 - 25 Zoltan Varga < vargaz @ freemail . hu >
2007 - 07 - 15 Atsushi Enomoto < atsushi @ ximian . com >
[ profiler ] Don', "'", 't call buffer_add_objid in DFS2 . Speeds up some cases by orders of magnitude .
[ Coop ] Convert performance counters and performance counter categories . ( )
2005 - 04 - 22 Chris Toshok < toshok @ ximian . com >
2009 - 01 - 31 Rodrigo Kumpera < rkumpera @ novell . com >
2004 - 03 - 03 Ben Maurer < bmaurer @ users . sourceforge . net >
2003 - 08 - 15 Daniel Morgan < danmorg @ sc . rr . com >
[ interp ] Remove virtual call duplicated code
refactoring
2002 - 05 - 10 Gaurav Vaish < gvaish_mono @ lycos . com >
[ aot ] Avoid a slow path in mono_jit_info_table_find ( ) .
2008 - 06 - 13 Marek Safar < marek . safar @ gmail . com >
from head
2010 - 01 - 08 Rodrigo Kumpera < rkumpera @ novell . com >
2009 - 04 - 23 Rodrigo Kumpera < rkumpera @ novell . com >
New test .
[ build ] Add ItemDefinitionGroup basic handling
* TypeTest . cs : Add switch to display method definitions and references .
2004 - 04 - 26 Marek Safar < marek . safar @ seznam . cz >
2005 - 01 - 13 Chris Toshok < toshok @ ximian . com >
[ arm ] Fix crashes on ios9 by using 16kb trampoline pages .
2007 - 10 - 15 Juraj Skripsky < js @ hotfeet . ch >
Remove redundant
[ monodoc ] Add missing legacy GetTitle method
2009 - 04 - 21 Atsushi Enomoto < atsushi @ ximian . com >
2004 - 05 - 13 Bernie Solomon < bernard @ ugsolutions . com >
2004 - 06 - 08 Marek Safar < marek . safar @ seznam . cz >
* ScrollableControl . cs : Calculate the maximum and change vars
[ aprofutil ] Update the tool name
2009 - 10 - 18 Rodrigo Kumpera < rkumpera @ novell . com >
2009 - 07 - 24 Marek Safar < marek . safar @ gmail . com >
[ mini ] Cleanup MONO_MMAP_32BIT . ( )
Fix s390x unwinding and remove need for backchain pointer ( )
2009 - 05 - 03 Marek Habersack < mhabersack @ novell . com >
[ jit ] Precompute a bit more information for delegate trampolines to speed them up .
A test for
2004 - 08 - 31 Martin Baulig < martin @ ximian . com >
[ corlib ] Fix TimeZoneInfo . Local when / usr / share / zoneinfo is a symlink ( )
2008 - 05 - 06 Marek Safar < marek . safar @ gmail . com >
[ msvc ] Update csproj files
2002 - 09 - 23 Dietmar Maurer < dietmar @ ximian . com >
2010 - 05 - 25 Atsushi Enomoto < atsushi @ ximian . com >
2003 - 03 - 20 Atsushi Enomoto < ginga @ kit . hi - ho . ne . jp >
[ runtime ] Remove some dead code .
2007 - 12 - 20 Jonathan Pobst < monkey @ jpobst . com >
2002 - 10 - 19 Tim Coleman < tim @ timcoleman . com >
2003 - 07 - 24 cesar lopez nataren < cesar @ ciencias . unam . mx >
2008 - 01 - 17 Zoltan Varga < vargaz @ gmail . com >
[ sdb ] Fix the build .
[ wasm ] Enable Mono . Native . ( )
[ corlib ] Fix UnicodeEncoding GetMaxByteCount and GetMaxCharCount to return the same values as . net .
2009 - 05 - 16 Carlos Alberto Cortez < calberto . cortez @ gmail . com >
2008 - 01 - 30 Marek Safar < marek . safar @ gmail . com >
Bump bockbuild
2007 - 08 - 02 Jonathan Pobst < monkey @ jpobst . com >
[ System . Net . Http ] Remove dead files
2004 - 11 - 01 Atsushi Enomoto < atsushi @ ximian . com >
[ sgen ] Remove CARDTABLE_STATS
2007 - 02 - 20 Atsushi Enomoto < atsushi @ ximian . com >
normalize line endings
2006 - 01 - 22 Zoltan Varga < vargaz @ gmail . com >
Fix wrong string tests .
[ corlib ] Fixes mobile build
2008 - 01 - 21 Marek Habersack < mhabersack @ novell . com >
2004 - 11 - 05 Atsushi Enomoto < atsushi @ ximian . com >
Revert patches to ProxyAttribute ( oops ) and add them to
[ corlib ] Fix culture info error message . Fixes
2001 - 09 - 19 Dietmar Maurer < dietmar @ ximian . com >
[ runtime ] Get rid of the create_delegate_trampoline_data ( ) helper function which only has one caller .
* ilasm . exe . sources : Add ExternTypeRefInst to the build .
2002 - 04 - 17 Rachel Hestilow < hestilow @ ximian . com >
2002 - 12 - 11 Jeffrey Stedfast < fejj @ ximian . com >
Revert
2002 - 03 - 10 Duncan Mak < duncan @ ximian . com >
- Now throws a nice error message when no X display could be opened
typo fix
Merge pull request from lambdageek / bug - 42584
[ arm ] Fix a regression introduced by : ( )
* RichBoxTest . cs : Disabled ReadOnly test until I can test it on linux .
2008 - 04 - 13 George Giolfan < georgegiolfan @ yahoo . com >
[ aprofutil ] Update the tool name
Final file changes
- Added comment
Nullable : Add msbuild project file
2002 - 10 - 02 Gaurav Vaish < gvaish @ iitk . ac . in >
2008 - 08 - 12 Jonathan Pobst < monkey @ jpobst . com >
2009 - 10 - 19 Atsushi Enomoto < atsushi @ ximian . com >
Hopefully fix the windows build .
[ wasm ] [ debugger ] Hash messages and sessions directly ( )
[ corlib ] : Add a workaround for redirected StandardInput writes . Fixes
[ build ] Update formatting to mono - api - profiler . ( )
[ jit ] Remove unused LLVM_MONO_BRANCH define ( )
[ wasm ] Small fixes . ( )
we pass those
2008 - 28 - 15 Andreas Nahr < ClassDevelopment @ A - SoftTech . com >
Add new test for bug
new tests
Add new files
* SoapAttribute . cs , SoapFieldAttribute . cs , SoapMethodAttribute . cs ,
[ interp ] Fix a regression introduced by : ( )
2009 - 04 - 11 Sebastien Pouliot < sebastien @ ximian . com >
2008 - 09 - 07 Jonathan Pobst < monkey @ jpobst . com >
2002 - 09 - 14 Martin Baulig < martin @ gnome . org >
update
 cvsignore
[ Reflection ] Rename variable
new error tests
2003 - 10 - 23 Zoltan Varga < vargaz @ freemail . hu >
2007 - 02 - 12 Igor Zelmanovich < igorz @ mainsoft . com >
2010 - 03 - 19 Jérémie Laval < jeremie . laval @ gmail . com >
undo r54467 changes
2003 - 10 - 25 Zoltan Varga < vargaz @ freemail . hu >
2007 - 10 - 13 Chris Toshok < toshok @ ximian . com >
[ runtime ] mono_optimize_branches the longer breaks with OP_IL_SEQ_POINT .
Merge pull request from akoeplinger / fix - processtest - race
Transaction now has limited support for PromotableSinglePhaseEnlistment
2004 - 05 - 26 Martin Baulig < martin @ ximian . com >
2004 - 11 - 05 Zoltan Varga < vargaz @ freemail . hu >
[ sgen ] Fix build on C99 .
[ System ] Switch DNS tests to a more stable domain / IP
ChangeLog : Updated ChangeLog .
Mon Oct 14 14 : 45 : 37 CEST 2008 Paolo Molaro < lupus @ ximian . com >
2010 - 04 - 21 Carlos Alberto Cortez < calberto . cortez @ gmail . com >
2010 - 04 - 10 Atsushi Enomoto < atsushi @ ximian . com >
Revert " [ Fix ] Fix the default binder when selecting params methods . "
2005 - 03 - 04 Sebastien Pouliot < sebastien @ ximian . com >
[ runtime ] Fix the build .
2006 - 02 - 26 Zoltan Varga < vargaz @ gmail . com >
2003 - 07 - 05 Dick Porter < dick @ ximian . com >
2010 - 02 - 26 Carlos Alberto Cortez < calberto . cortez @ gmail . com >
[ System . Windows . Forms ] Ignore non - visible controls when computing ScrollableControl canvas size ( )
implement IsDirty ( ) ( incomplete of course ) , MarkDirty ( ) and DisableMarkDirty . Fix tests .
[ corlib ] Restores ConcurrentDictionary serialization support .
syncing
2010 - 04 - 02 Atsushi Enomoto < atsushi @ ximian . com >
oops commited some debug code .
2005 - 03 - 10 Marek Safar < marek . safar @ seznam . cz >
- Part of the previus checkin , missed these files
2009 - 05 - 02 Rodrigo Kumpera < rkumpera @ novell . com >
* LogConverter . cs : stubbed
Tue May 15 16 : 30 : 34 CEST 2002 Paolo Molaro < lupus @ ximian . com >
ChangeLog : Updated ChangeLog .
[ monkeydoc ] Early return if no proper fallback index was found in MatchNode
Flush
2006 - 03 - 13 Gonzalo Paniagua Javier < gonzalo @ ximian . com >
[ sgen ] Run DynArray into strongly types variants making the code cleaner and safer .
New test .
Tue Aug 4 . 0 : 53 : 17 CEST 2003 Paolo Molaro < lupus @ ximian . com >
Oops
typo fix
[ runtime ] Add - - assembly - loader = [ strict , legacy ] runtime argument
2007 - 05 - 06 Marek Safar < marek . safar @ gmail . com >
* UpDownTest . cs : Only apply ComVisible attribute to FilterLevel
2004 - 11 - 30 Zoltan Varga < vargaz @ freemail . hu >
2004 - 04 - 02 Bernie Solomon < bernard @ ugsolutions . com >
2006 - 07 - 26 Zoltan Varga < vargaz @ gmail . com >
2007 - 08 - 17 Carlos Alberto Cortez < calberto . cortez @ gmail . com >
2005 - 10 - 05 Peter Dennis Bartok < pbartok @ novell . com >
Fix the previous marshalling change so it only applies to chars , not structures which contain char fields .
undo r54467 changes
Replace scp for rsync , updating the web page was getting slow
[ runtime ] Fix a warning .
flush
2001 - 11 - 05 Dick Porter < dick @ ximian . com >
improve tests
Revert " [ runtime ] Fix the computation of token sizes in ppdb files , the PDB stream has an additional table which contains the token sizes for the metadata tables referenced by the ppdb tables . ( ) "
2010 - 04 - 03 Zoltan Varga < vargaz @ gmail . com >
[ corlib ] Update BCL Linked Size
Revert " [ Mono . Security ] Stub out a lot of classes to make NuGet build with mono .
[ interp ] Use frame_addr as base pointer in more places ( )
* SoapFormatter . cs : Use IndexOf instead of Leave for
[ mini ] Use mono_debug_method_lookup_location ( ) instead of mono_debug_symfile_lookup_location ( ) , the latter doesn', "'", 't support portable pdb files . Use mono_debug_free_source_location ( ) for freeing . Fixes . ( )
2001 - 12 - 21 Miguel de Icaza < miguel @ ximian . com >
2004 - 09 - 19 Zoltan Varga < vargaz @ freemail . hu >
refactoring
Bump bockbuild
2001 - 11 - 24 Miguel de Icaza < miguel @ ximian . com >
2004 - 03 - 04 Andreas Nahr < ClassDevelopment @ A - SoftTech . com >
[ jit ] Precompute a bit more information for delegate trampolines to speed them up .
2008 - 01 - 22 Sebastien Pouliot < sebastien @ ximian . com >
2005 - 12 - 21 Martin Baulig < martin @ ximian . com >
Forgot to commit this
2003 - 04 - 25 Sebastien Pouliot < spouliot @ videotron . ca >
[ jit ] Run static cctors for inlined methods , the cctor might not have ran in AOT mode . ( )
Fix the build .
2006 - 08 - 30 Alexander Olk < alex . olk @ googlemail . com >
2006 - 09 - 15 Gonzalo Paniagua Javier < gonzalo @ ximian . com >
[ Tailcall ] Proper fix for F # regression . ( )
more docu
2010 - 07 - 02 Veerapuram Varadhan < vvaradhan @ novell . com >
column name and ordinal fix . . . tested on 10 . 1
[ netcore ] Use mono_debug_method_lookup_location ( ) ( )
bootstrap fix
Add CSharpLib docs
Replaced all assembly references with constants in Consts . cs .
Bump API snapshot submodule
2007 - 03 - 13 Nagappan A < anagappan @ novell . com >
2008 - 06 - 20 Zoltan Varga < vargaz @ gmail . com >
Always restore extension methods candidates after each run
2005 - 06 - 14 Zoltan Varga < vargaz @ freemail . hu >
2003 - 01 - 14 Ben Maurer < bmaurer @ users . sourceforge . net >
Make sure configuration is not null before using it .
Fri Nov 19 17 : 35 : 49 CET 2002 Paolo Molaro < lupus @ ximian . com >
2005 - 06 - 08 Zoltan Varga < vargaz @ freemail . hu >
[ msvc ] Update csproj files
2007 - 09 - 15 Jonathan Pobst < monkey @ jpobst . com >
[ System . Net . Http ] Add IVT to Xamarin . * assemblies and update Mac
* * * empty log message * * *
2009 - 06 - 06 Ivan N . Zlatev < contact @ i - nz . net >
Fix the build .
oops , fix the build
[ runtime ] Fix the build .
Added MIT license to System . dll
[ jit ] Transition the SIMD code to use the MONO_CPU_ . . . set of flags instead of the SIMD_VERSION_ . . . flags . ( )
2007 - 11 - 07 Zoltan Varga < vargaz @ gmail . com >
[ bcl ] Fix the build .
Add System . UriTypeConverter ( 2 . 0 ) with unit / CAS tests , also add CAS / unit tests for System . ComponentModel . Design . Serialization . InstanceDescriptor
2003 - 03 - 22 Zoltan Varga < vargaz @ freemail . hu >
2004 - 04 - 17 Atsushi Enomoto < atsushi @ ximian . com >
updates from grammar changes
Don', "'", 't build aot symbols on make dist
[ utils ] Add foreach variant that supports filtering the list as you go .
[ llvm ] Fix the loadable llvm build .
2004 - 04 - 16 Dick Porter < dick @ ximian . com >
Add new test
ChangeLog : Updated ChangeLog .
[ System . Private . CoreLib ] Add Nullable runtime support
Fix
Populate properties of AssemblyLoadStart / Stop events ( dotnet / coreclr )
[ debugger ] Fix a warning .
Dear God , make me recompile my code after removing debug * before * I check in
2005 - 10 - 07 Zoltan Varga < vargaz @ gmail . com >
[ runtime ] Register get gsharedvt arg trampolines
Fix handle warning .
2004 - 03 - 22 Sebastien Pouliot < sebastien @ ximian . com >
[ profiler ] Avoid emitting seq points inside alloc wrappers , they contain critical regions , and if thread self suspend in the seq point handler , it cannot be suspended by other threads . Fixes .
2009 - 06 - 30 Zoltan Varga < vargaz @ gmail . com >
2008 - 01 - 07 Marek Habersack < mhabersack @ novell . com >
[ dist ] Ensure gensources . cs is included in tarballs ( )
2007 - 11 - 06 Marek Safar < marek . safar @ gmail . com >
[ xbuild ] Evaluate Import only if its condition evaluates to true .
Add', "'", 'mdoc update - ecma - xml', "'", 'command .
[ build ] Fix a test .
2005 - 01 - 06 Martin Baulig < martin @ ximian . com >
2003 - 10 - 03 Ben Maurer < bmaurer @ users . sourceforge . net >
fix erroneous', "'", 'Completed', "'", 'label in legend .
* statement . cs ( Statement . Clone ) : Move special case code to Block . CloneTo .
2010 - 06 - 17 Zoltan Varga < vargaz @ gmail . com >
2005 - 09 - 04 Zoltan Varga < vargaz @ freemail . hu >
[ docs ] Re - add gacutil ( )
2008 - 09 - 01 Andreas Nahr < ClassDevelopment @ A - SoftTech . com >
2002 - 06 - 11 Miguel de Icaza < miguel @ ximian . com >
* TreeView . cs : Use dashed lines to connect nodes . Use the
Add comment
* TabControl . cs : Use dashed lines to connect nodes . Use the
2004 - 07 - 14 Miguel de Icaza < miguel @ ximian . com >
fix 1 . x build
[ jit ] Add a new ICALL_ADDR_CALL patch type which is only used on calls , since ICALL_ADDR might be used without a call .
[ corlib ] Add CreateTypeInfo method
[ coop ] Add a handle frame in construct_culture ( )
[ corlib ] Fix UnicodeEncoding GetMaxByteCount and GetMaxCharCount to return the same values as . net .
nuke PropertyGridEntry . cs
2004 - 04 - 16 Bernie Solomon < bernard @ ugsolutions . com >
2009 - 10 - 24 Rodrigo Kumpera < rkumpera @ novell . com >
2006 - 10 - 06 Dick Porter < dick @ ximian . com >
Temporary checked - in repackaged admin app . TODO : fix repackaging issue , or add support for so - called " flat webapp " inside GH build system .
2008 - 04 - 23 Jonathan Chambers < joncham @ gmail . com >
* SerializationCodeGenerator . cs : Minor fixes .
* * * empty log message * * *
Tweak the Unreal profile
Extracts public IThreadPoolWorkItem interface to shared CoreLib ( )
Bump ikvm
[ runtime ] Add support for using COM types from corlib .
2009 - 03 - 20 Bill Holmes < billholmes54 @ gmail . com >
2004 - 09 - 26 Atsushi Enomoto < atsushi @ ximian . com >
[ llvm ] Revert docs for mono_profiler_set_sample_mode ( ) .
[ Coop ] ves_icall_RuntimeFieldInfo_GetRawConstantValue . ( )
[ Mono . Debugger . Soft ] Catching AssemblyMirror in AppDomainMirror and TypeMirror in AssemblyMirror
Thu Sep 4 16 : 34 : 49 CEST 2007 Paolo Molaro < lupus @ ximian . com >
2010 - 04 - 16 Miguel de Icaza < miguel @ novell . com >
2004 - 09 - 26 Cesar Lopez Nataren < cesar @ ciencias . unam . mx >
[ runtime ] Make the code in process_call ( ) a bit .
[ corlib ] Match closer ExecutionContextSwitcher methods
2007 - 10 - 14 Sebastien Pouliot < sebastien @ ximian . com >
2005 - 03 - 23 Martin Baulig < martin @ ximian . com >
2006 - 05 - 03 Marek Safar < marek . safar @ seznam . cz >
merge r108940 - 108941
[ csproj ] Update project files
Add documentatin from Nant
2008 - 05 - 17 Atsushi Enomoto < atsushi @ ximian . com >
2008 - 03 - 25 Atsushi Enomoto < atsushi @ ximian . com >
Merge pull request from jonpryor / jonp - MonoThread - accessors
2007 - 08 - 28 Jb Evain < jbevain @ novell . com >
2008 - 01 - 29 Zoltan Varga < vargaz @ gmail . com >
see last note
* TreeView . cs : Use dashed lines to connect nodes . Use the
2004 - 02 - 17 Atsushi Enomoto < atsushi @ ximian . com >
[ sgen ] Fix build on C99 .
Add unit tests specific for RC2CryptoServiceProvider and TripleDESCryptoServiceProvider implementations ( i . e . not the algorithms )
2006 - 09 - 17 Chris Toshok < toshok @ ximian . com >
Handle Bool w . r . t Logical operators
[ runtime ] Use MonoError for mono_class_vtable_full and mono_runtime_class_init_full .
More error message .
Merge pull request from esdrubal / fileinfo_moveto
Add new files
Flush
[ profiler ] Type error to a underlying worker thread .
2006 - 09 - 29 Igor Zelmanovich < igorz @ mainsoft . com >
2010 - 04 - 26 Zoltan Varga < vargaz @ gmail . com >
[ corlib ] Fix culture info error message . Fixes
2002 - 01 - 07 Tim Coleman < tim @ timcoleman . com >
2005 - 12 - 15 Sebastien Pouliot < sebastien @ ximian . com >
* AppDomainSetupTest . cs : Changed default formatter sink to BinaryFormatterSink
Bump ikvm
2008 - 12 - 20 Carlos Alberto Cortez < calberto . cortez @ gmail . com >
[ runtime ] Fix a warning .
Compile fix .
[ mcs ] Add missing vector comments ( )
Add more files
* cs1502 . cs , SoapFieldAttribute . cs , SoapMethodAttribute . cs ,
2005 - 02 - 13 Peter Bartok < pbartok @ novell . com >
A tests for bug
2004 - 12 - 06 Duncan Mak < duncan @ ximian . com >
[ msvc ] Fix Mono . Native . ( )
[ jit ] Remove the support for MONO_ARCH_NOTIFY_PENDING_EXC , it only works on amd64 , and its not worth the complexity .
[ runtime ] Fix the build .
[ 642437 ] Nested types have to check type arguments only and not full class
* DataColumnCollectionTest . cs : Added test for Add ( String ) with null
* XmlReflectionImporter . cs : Type specified with XmlIncludeAttribute don', "'", 't
[ netcore ] Clean up some warnings
[ runtime ] Fix the build .
2002 - 08 - 11 Martin Baulig < martin @ gnome . org >
[ jit ] Remove some dead code / includes .
[ metadata ] Update libtiff ( )
Test for the arm encoding bug .
2007 - 04 - 29 Zoltan Varga < vargaz @ gmail . com >
[ mini ] Implement Arm intrinsics : ArmBase , Crc32 ( )
refactoring
crlf fix
[ ci ] Remove - f option from genhtml during code coverage
* * * * Merged r41636 from MCS * * * *
2006 - 02 - 17 Chris Toshok < toshok @ ximian . com >
[ mcs ] Emit correct token for inherited nested non - generic types from unbound types
Love the Makefile
2006 - 04 - 04 Miguel de Icaza < miguel @ novell . com >
2008 - 10 - 06 Zoltan Varga < vargaz @ gmail . com >
* * * empty log message * * *
2007 - 08 - 13 Jonathan Pobst < monkey @ jpobst . com >
2002 - 09 - 11 Martin Baulig < martin @ gnome . org >
2006 - 01 - 05 Zoltan Varga < vargaz @ gmail . com >
[ metadata ] Don', "'", 't show the deprecation notice for minimal or quiet logging
2006 - 03 - 20 Chris Toshok < toshok @ ximian . com >
2008 - 03 - 06 Jb Evain < jbevain @ novell . com >
2007 - 12 - 31 Atsushi Enomoto < atsushi @ ximian . com >
2005 - 03 - 04 Duncan Mak < duncan @ novell . com >
another test
2004 - 09 - 30 Atsushi Enomoto < atsushi @ ximian . com >
[ corlib ] Mark the SRE types with required = false so they can be linked out . ( )
[ io - layer ] Remove unnecessary locking of _WAPI_SHARED_SEM_FILESHARE
[ profiler ] Disable fast support for pointer ( )
2003 - 04 - 10 Andreas Nahr < ClassDevelopment @ A - SoftTech . com >
Use correct result type for chained folded enum subtraction
2009 - 05 - 02 Gonzalo Paniagua Javier < gonzalo @ novell . com >
2010 - 04 - 21 Gonzalo Paniagua Javier < gonzalo @ novell . com >
2007 - 10 - 22 Alp Toker < alp @ atoker . com >
2009 - 05 - 31 Rodrigo Kumpera < rkumpera @ novell . com >
2009 - 04 - 27 Zoltan Varga < vargaz @ gmail . com >
2009 - 03 - 08 Sebastien Pouliot < sebastien @ ximian . com >
2003 - 12 - 04 Ben Maurer < bmaurer @ users . sourceforge . net >
[ reflection ] Use mono_debug_method_lookup_location ( ) instead of mono_debug_symfile_lookup_location ( ) , the latter doesn', "'", 't support portable pdb files . Use mono_debug_free_source_location ( ) for freeing . Fixes . ( )
More tests .
[ corlib ] When finalizing the ECMA key , don', "'", 't produce a public key . Fixes
2003 - 11 - 15 Atsushi Enomoto < ginga @ kit . hi - ho . ne . jp >
2006 - 10 - 30 Chris Toshok < toshok @ ximian . com >
2008 - 06 - 19 Zoltan Varga < vargaz @ gmail . com >
* DateTimePicker . cs : Fixed a NRE if CustomFormat was null . Handles the
2009 - 05 - 11 Rodrigo Kumpera < rkumpera @ novell . com >
Fix MonoBtlsX509StoreManager machine deserialization
* NumberFormatter . cs : Fixed compiler warning .
[ interp ] Remove unused dead code .
revert
put in the right namespace
* PlatformID . cs ( dis_method_list ) : Set TdsVersion to the version
Wed Feb 8 18 : 50 : 47 CET 2006 Paolo Molaro < lupus @ ximian . com >
Merge pull request from kumpera / fix - the - build
2006 - 08 - 30 Mike Kestner < mkestner @ novell . com >
2002 - 10 - 09 Brian Ritchie < brianlritchie @ hotmail . com >
2008 - 04 - 15 Jb Evain < jbevain @ novell . com >
2008 - 03 - 07 George Giolfan < georgegiolfan @ yahoo . com >
oops
Add ExceptionDispatchInfo . SetCurrentStackTrace ( dotnet / coreclr )
2009 - 04 - 29 Marek Habersack < mhabersack @ novell . com >
[ corlib ] Import System . IO . Stream / BufferedStream from CoreFX ( )
2010 - 01 - 15 Atsushi Enomoto < atsushi @ ximian . com >
2003 - 04 - 16 Ivan Figueiredo Jr . < fxjrlists @ yahoo . com . br >
2003 - 11 - 13 Miguel de Icaza < miguel @ ximian . com >
* ILParser . jay : Create strings from byte arrays . Fix parsing of
Merge pull request from akoeplinger / fix - mono - free - alloc2
[ corlib ] First specific culture returned by enumerator was not fully initialized . Fixes
2008 - 10 - 09 Jb Evain < jbevain @ novell . com >
2003 - 07 - 08 Atsushi Enomoto < ginga @ kit . hi - ho . ne . jp >
[ System . Windows . Forms ] Incomplete byte ranges parsing . Fixes
[ System ] Add IPv6 support to FtpWebRequest
2008 - 07 - 03 Zoltan Varga < vargaz @ gmail . com >
Use _vscprintf ( ) in mono_bitset_count ( ) .
[ wasm ] Small fixes around .
[ acceptance - tests ] Bump coreclr to pick up 3b2969dffe191b03f1ac85a9c1e62417c9b7833b :
[ acceptance - tests ] Bump coreclr to pick up 3b2969dffe191b03f1ac85a9c1e62417c9b7833b :
Add DecimalFormatterTest . cs )
* ILParser . jay : Use new sentinel operator . Create valuetypes .
* ILParser . jay : Create external method references
[ sgen ] Separate binary protocol entries for concurrent update and finish .
2004 - 09 - 09 Martin Baulig < martin @ ximian . com >
[ System ] Add IPv6 support to FtpWebRequest
2005 - 03 - 30 Gonzalo Paniagua Javier < gonzalo @ ximian . com >
2003 - 07 - 13 Atsushi Enomoto < ginga @ kit . hi - ho . ne . jp >
[ runtime ] Fail classes that have a field with a bad underlying type
Minor orbis fix in System . Net
[ System . ComponentModel . Composition ] Updated Makefile thisdir .
[ Coop ] Convert dynamic - image . c . ( )
Wed Dec 13 16 : 50 : 14 CEST 2002 Paolo Molaro < lupus @ ximian . com >
2002 - 01 - 16 Nick Drochak < ndrochak @ gol . com >
2008 - 06 - 17 Atsushi Enomoto < atsushi @ ximian . com >
2002 - 02 - 22 Nick Drochak < ndrochak @ gol . com >
[ runtime ] Enable unwind_backtrace on ARM / linux ( )
Merge pull request from lateralusX / lateralusX / fix - win - fullaot - runtime - tests - toggleref
[ xbuild ] Evaluate Import only if its condition evaluates to true .
2005 - 12 - 16 Carlos Alberto Cortez < calberto . cortez @ gmail . com >
2004 - 05 - 01 Gonzalo Paniagua Javier < gonzalo @ ximian . com >
[ corlib ] Restores ConcurrentDictionary serialization support .
New test .
Bump bockbuild
[ Mono . Debugger . Soft ] Catching AssemblyMirror in AppDomainMirror and TypeMirror in AssemblyMirror
2009 - 04 - 23 Geoff Norton < gnorton @ novell . com >
2010 - 02 - 10 Atsushi Enomoto < atsushi @ ximian . com >
2002 - 05 - 12 Martin Baulig < martin @ gnome . org >
[ sgen ] Fix small localized warnings .
2008 - 07 - 23 Zoltan Varga < vargaz @ gmail . com >
removes fix
2009 - 05 - 15 Gonzalo Paniagua Javier < gonzalo @ novell . com >
Add new files
[ corlib ] Interning of utc literals in TimeZoneInfo conversion . Fixes
* main . c ( dis_method_list ) : Oops . Check method_context for null .
[ Coop ] Convert dynamic - image . c . ( )
Fix ToDateTime ( ) on Windows ( )
2002 - 03 - 17 Dietmar Maurer < dietmar @ ximian . com >
System . Drawing . vmwcsproj : Added SystemIcons . jvm . cs , excluded SystemIcons . cs
Merge pull request from akoeplinger / pupitetris - Bug - 35635
[ runtime ] Avoid allocating interface offsets multiple times . Fixes
2009 - 01 - 30 Zoltan Varga < vargaz @ gmail . com >
2010 - 06 - 10 Atsushi Enomoto < atsushi @ ximian . com >
new tests
2009 - 02 - 25 Rodrigo Kumpera < rkumpera @ novell . com >
2002 - 11 - 30 Tim Coleman < tim @ timcoleman . com >
2005 - 07 - 18 Alexander Olk < alex . olk @ googlemail . com >
[ Coop ] ves_icall_RuntimeFieldInfo_GetRawConstantValue . ( )
2007 - 08 - 14 Atsushi Enomoto < atsushi @ ximian . com >
[ runtime ] Add MONO_ARCH_HAVE_SIGCTX_TO_MONOCTX on every PowerPC configuration on Linux
2009 - 10 - 06 Atsushi Enomoto < atsushi @ ximian . com >
we pass those
2007 - 08 - 26 Marek Sieradzki < marek . sieradzki @ gmail . com >
[ llvm ] Remove some dead code / includes .
[ mini ] Remove CARDTABLE_STATS
[ threads ] Add intrinsics support for context operations to restart
2004 - 09 - 14 Zoltan Varga < vargaz @ freemail . hu >
Small improvement
use style_attach
[ Microsoft . Build ] Stub out a lot of classes to make NuGet build with mono .
2007 - 05 - 29 Jonathan Pobst < monkey @ jpobst . com >
2009 - 11 - 25 Zoltan Varga < vargaz @ gmail . com >
Merge pull request from ch5oh / master
2008 - 08 - 03 Zoltan Varga < vargaz @ gmail . com >
2007 - 10 - 02 Dick Porter < dick @ ximian . com >
2003 - 12 - 04 Ben Maurer < bmaurer @ users . sourceforge . net >
2008 - 09 - 03 Miguel de Icaza < miguel @ novell . com >
2008 - 10 - 04 Andreas Nahr < ClassDevelopment @ A - SoftTech . com >
2005 - 07 - 23 Zoltan Varga < vargaz @ freemail . hu >
Merge pull request from akoeplinger / fix - ipv6 - teredo
Recover file
[ Tailcall ] Proper fix for F # regression . ( )
* ScrollableControl . cs : Calculate the maximum and change vars
[ ci ] Fix mozroots URL ( )
2010 - 05 - 31 Sebastien Pouliot < sebastien @ ximian . com >
2008 - 04 - 27 Jonathan Pobst < monkey @ jpobst . com >
[ tests ] ManualResetEventSlim . SpinCount depends on the number of CPU / cores and we can', "'", 't assume everything as multiple CPU / cores [ ]
2006 - 02 - 11 Atsushi Enomoto < atsushi @ ximian . com >
* Makefile : add gacutil to the build
Merge pull request from kumpera / wasm_fix_a_few_issues
Up
2005 - 06 - 16 Zoltan Varga < vargaz @ freemail . hu >
several ComboBox back
2009 - 07 - 10 Marek Safar < marek . safar @ gmail . com >
Merge pull request from rolfbjarne / fix - assert - in - poll_event_wait - on - WSAENOTSOCK - on - windows
2002 - 09 - 10 Tim Coleman < tim @ timcoleman . com >
[ sgen ] Use our mutex abstraction instead of pthreads directly .
[ jit ] Remove the MONO_INIT_VARINFO macro , its only used in one place . ( )
[ corlib ] Fix UnicodeEncoding GetMaxByteCount and GetMaxCharCount to return the same values as . net .
2005 - 09 - 02 Gonzalo Paniagua Javier < gonzalo @ ximian . com >
* * * empty log message * * *
* HttpResponse . cs : Temp source files should have extension " . vb " .
[ metadata ] Add Mono . Cecil . * . Has * checks when comparing methods / types
[ runtime ] Fixed unused ` count ` variable warning by moving some code around .
Bump to mono / corefx @ e79cf5b
Changes sourceStream . Read to read in a while loop
2007 - 11 - 05 Carlos Alberto Cortez < calberto . cortez @ gmail . com >
2004 - 01 - 08 Nick Drochak < ndrochak @ ieee . org >
Moving mess
2005 - 09 - 01 Atsushi Enomoto < atsushi @ ximian . com >
[ sgen ] Fix a toggleref array resize bug .
Added new files
Improve tests
2009 - 04 - 06 Marek Safar < marek . safar @ gmail . com >
2006 - 07 - 23 Marek Sieradzki < marek . sieradzki @ gmail . com >
2003 - 12 - 22 Sebastien Pouliot < spouliot @ videotron . ca >
2007 - 01 - 19 Carlos Alberto Cortez < calberto . cortez @ gmail . com >
2010 - 07 - 06 Carlos Alberto Cortez < calberto . cortez @ gmail . com >
[ cxx ] [ x86 ] int / ptr casts . ( )
Build one
2005 - 02 - 26 Atsushi Enomoto < atsushi @ ximian . com >
[ runtime ] Add Mono . Cecil . * . Has * checks when comparing methods / types
2009 - 10 - 21 Gonzalo Paniagua Javier < gonzalo @ novell . com >
* * * empty log message * * *
interp / interp . c : In function', "'", 'interp_entry . constprop', "'", ':
New tests .
2009 - 05 - 04 Zoltan Varga < vargaz @ gmail . com >
2009 - 02 - 04 Zoltan Varga < vargaz @ gmail . com >
[ threadpool ] Harder the custom attribute code against common type load errors .
* HttpResponse . cs : Temp source files should have extension " . vb " .
2004 - 12 - 16 Atsushi Enomoto < atsushi @ ximian . com >
q
2005 - 09 - 15 Carlos Alberto Cortez < calberto . cortez @ gmail . com >
[ profiler ] Add myself .
* runtime - cil . c ( get_encoded_user_string ) : use g_strescape instead of
Checks unittest to the snippets examples
[ mcs ] Reset doc reader state after expression bodied property . Fixes
[ wasm ] [ debugger ] Add support for surfacing inherited members ( )
[ corlib ] Fix UnicodeEncoding GetMaxByteCount and GetMaxCharCount to return the same values as . net .
use style_attach
[ jit ] Remove some dead code / includes .
2006 - 07 - 06 Geoff Norton < gnorton @ customerdna . com >
2001 - 12 - 21 Nick Drochak < ndrochak @ gol . com >
Fixed typo .
2005 - 11 - 29 Atsushi Enomoto < atsushi @ ximian . com >
2008 - 09 - 29 Carlos Alberto Cortez < calberto . cortez @ gmail . com >
2002 - 05 - Add Gonzalo Paniagua Javier < gonzalo @ ximian . com >
[ llvm ] Disable the locking in set_breakpoint ( ) a bit .
2005 - 12 - 20 Sebastien Pouliot < sebastien @ ximian . com >
q
[ runtime ] Fix Process . TotalProcessorTime becoming negative . Fixes .
Add new files
New tests .
[ interp ] Run off Boehm on Windows x64 gcc build
2008 - 02 - 11 Zoltan Varga < vargaz @ gmail . com >
2005 - 01 - 30 Martin Baulig < martin @ ximian . com >
New tests .
Recover file
[ jit ] Avoid pushing the lmf in inlined methods . Allow the variable holding the lmf address to be allocated to a global register .
2002 - 01 - 04 Sebastien Pouliot < spouliot @ videotron . ca >
Fix some counters under 32bits .
2004 - 10 - 18 Zoltan Varga < vargaz @ freemail . hu >
[ Mono . Security ] : add API to let MonoTlsProvider provider a custom system certificate validator .
2008 - 06 - 02 Sebastien Pouliot < sebastien @ ximian . com >
more docu
2005 - 09 - 01 Sebastien Pouliot < sebastien @ ximian . com >
2003 - 12 - 28 Patrik Torstensson
no longer needed
[ sgen ] Precisely log the size of the promoted objects during minors
* EventLogTest . cs : Use a dedicated temp directory for testing the
2008 - 08 - 06 Carlos Alberto Cortez < calberto . cortez @ gmail . com >
Test for .
2002 - 01 - 04 Ravi Pratap < ravi @ ximian . com >
2003 - 12 - 01 Dick Porter < dick @ ximian . com >
ChangeLog : Updated ChangeLog .
2005 - 12 - 11 Atsushi Enomoto < atsushi @ ximian . com >
2005 - 01 - 26 Sebastien Pouliot < sebastien @ ximian . com >
2003 - 04 - 17 Zoltan Varga < vargaz @ freemail . hu >
Remove doctools , so it displays a checkered bar during moving instead of constantly resizing the controls docked to it .
[ bcl ] Update BCL Linked Size
svn path = / trunk / mcs / ; revision = 51213
2010 - 04 - 29 Atsushi Enomoto < atsushi @ ximian . com >
2004 - 04 - 05 John Luke < john . luke @ gmail . com >
2005 - 04 - 11 Sureshkumar T < tsureshkumar @ novell . com >
Merge pull request from mayerwin / patch - 2
Fix
2002 - 08 - 28 Dick Porter < dick @ ximian . com >
2004 - 04 - 19 Gonzalo Paniagua Javier < gonzalo @ ximian . com >
New test .
2009 - 08 - 20 Atsushi Enomoto < atsushi @ ximian . com >
add MemberBind tests
2009 - 01 - 26 Marek Habersack < mhabersack @ novell . com >
2009 - 04 - 12 Carlos Alberto Cortez < calberto . cortez @ gmail . com >
[ xbuild ] Add ItemDefinitionGroup basic handling
* EventLog . cs : Minor fixes .
2002 - 09 - 16 Gaurav Vaish < gvaish_mono @ lycos . com >
Add documentatin submodule
[ jit ] Fix the freeing of AOT jump target lists . Fixes .
Change ARGS : System / ArrayTest . cs ( 36 )
2007 - 05 - 12 Amit Biswas < amit @ amitbiswas . com >
2007 - 08 - 25 Jb Evain < jbevain @ novell . com >
typo fix
2004 - 04 - 21 Dick Porter < dick @ ximian . com >
Fix s390x unwinding and remove need for backchain pointer ( )
[ mono ] Linking statically ICU shim on mono ( )
2003 - 10 - 25 Nick Drochak < ndrochak @ gol . com >
2006 - 02 - 09 Sebastien Pouliot < sebastien @ ximian . com >
2006 - 01 - 12 Peter Dennis Bartok < pbartok @ novell . com >
use Invariant culture for string manipulations
2005 - 04 - 20 Atsushi Enomoto < atsushi @ ximian . com >
[ System . XML ] Fix handling of global variables in XSLs ( )
2008 - 05 - 06 Jb Evain < jbevain @ novell . com >
2002 - 01 - 14 Dick Porter < dick @ ximian . com >
[ runtime ] Use mono_debug_method_lookup_location ( ) instead of mono_debug_symfile_lookup_location ( ) , the latter doesn', "'", 't support portable pdb files . Use mono_debug_free_source_location ( ) for freeing . Fixes . ( )
2004 - 12 - 08 Sebastien Pouliot < sebastien @ ximian . com >
[ configure ] Fix build when using musl libc .
[ runtime ] Move pending_exception from InternalThread to Thread to avoid crashing on domain unload .
* Page . cs : ResetEventValidationState call moved back to InternalProcessRequest
Attempt to fix
[ Coop ] ves_icall_RuntimeFieldInfo_GetRawConstantValue . ( )
* Convert . cs : In CreateMessageSink , ignore the url
2006 - 08 - 17 Marek Sieradzki < marek . sieradzki @ gmail . com >
[ threadpool - ms ] Fix incorrect use of mono_rand .
Add man page
move the System . Net 2 . 1 specific from System to System . Net
Improve
2004 - 01 - 02 Sebastien Pouliot < sebastien @ ximian . com >
2010 - 04 - 06 Marek Habersack < mhabersack @ novell . com >
[ runtime ] Make mono_value_box use MonoError
- Added Locale for ColorDialog ( de - DE )
[ interp ] Fix build
2003 - 01 - 10 Gonzalo Paniagua Javier < gonzalo @ ximian . com >
Tue May 10 16 : 58 : 52 CEST 2002 Paolo Molaro < lupus @ ximian . com >
Add enum support to build .
Fix IDE0029 ( use coalesce expression )
2009 - 02 - 11 Zoltan Varga < vargaz @ gmail . com >
2009 - 11 - 02 Jb Evain < jbevain @ novell . com >
[ runtime ] Remove some dead code .
Fix the build .
2005 - 04 - 12 Zoltan Varga < vargaz @ freemail . hu >
[ ppc ] Fix build
2005 - 10 - 18 Atsushi Enomoto < atsushi @ ximian . com >
System . Data . dll . sources : Added DbException . cs , SqlClientMetaDataCollectionNames . cs
[ monkeydoc ] Avoid extra copying of EcmaDesc
[ runtime ] Add - - assembly - loader = [ strict , legacy ] runtime argument
2008 - 05 - 10 Sebastien Pouliot < sebastien @ ximian . com >
Bump API snapshot submodule
* SqlParameter . cs ( ConvertToFrameworkType ) : Only wrap FormatException
2008 - 01 - 28 Marek Safar < marek . safar @ gmail . com >
2008 - 01 - 19 Jb Evain < jbevain @ novell . com >
2005 - 09 - 04 Martin Baulig < martin @ ximian . com >
[ asp . net ] Do nothing if null SessionStateStoreData is passed to SessionStateServerHandler . SetAndReleaseItemExclusive
Merge pull request from ntherning / winaot - tests - on - jenkins
* SerializationCodeGenerator . cs : Minor fixes .
[ profiler ] Add a _checked version for mono_class_from_name ( ) .
* PageCompiler . cs : Add Trace and TraceMode to framework initialize
2009 - 01 - 22 Gonzalo Paniagua Javier < gonzalo @ novell . com >
2004 - 01 - 31 Geoff Norton < gnorton @ customerdna . com >
[ jit ] Fix the check for calling mono_get_vtable_var ( ) , check for', "'", 'cfg - > method', "'", 'instead of', "'", 'method', "'", ', the latter is the inlined method . ( )
Flush
Up
2004 - 09 - 11 Martin Baulig < martin @ ximian . com >
2006 - 04 - 04 Chris Toshok < toshok @ ximian . com >
[ runtime ] Remove some dead code .
2010 - 06 - 24 Carlos Alberto Cortez < calberto . cortez @ gmail . com >
2004 - 08 - 04 Sebastien Pouliot < sebastien @ ximian . com >
[ tests ] Add deprecation warning
2003 - 09 - 20 Martin Baulig < martin @ ximian . com >
2002 - 10 - 26 Ville Palo < vi64pa @ koti . soon . fi >
2006 - 09 - 05 Jonathan Chambers < joncham @ gmail . com >
2005 - 04 - 22 Gonzalo Paniagua Javier < gonzalo @ ximian . com >
2008 - 12 - 16 Rodrigo Kumpera < rkumpera @ novell . com >
[ System . EnterpriseServices ] Rename method parameters to match . NET contract
2007 - 08 - 31 Zoltan Varga < vargaz @ gmail . com >
2008 - 03 - 29 Igor Zelmanovich < igorz @ mainsoft . com >
2007 - 01 - 05 Sebastien Pouliot < sebastien @ ximian . com >
2006 - 12 - 31 Atsushi Enomoto < atsushi @ ximian . com >
2010 - 03 - 27 Carlos Alberto Cortez < calberto . cortez @ gmail . com >
2010 - 01 - 18 Atsushi Enomoto < atsushi @ ximian . com >
Reworded and Reformatted the Readme
Added test for fd_set
[ netcore ] Enable netcore resolution logic for mono_assembly_byname_lookup ( )
* DataGrid . cs : call SetDataSource instead of CalcGridAreas in
2008 - 01 - 25 Jonathan Pobst < monkey @ jpobst . com >
use 32bpp bitmaps internally
update MS . Build test csproj .
[ runtime ] Make WaitForPendingFinalizers ( ) really wait for the finalizers added by a previous GC . Fixes .
- Added Locale to ColorDialog ( dotnet / coreclr )
[ System . Transactions ] Add stubs and visibility fixes for netstandard20 APIs
2009 - 07 - 04 Atsushi Enomoto < atsushi @ ximian . com >
Add test for Regex . Match with anchors on a substring .
Fri Jul 14 11 : 17 : 02 CEST 2002 Paolo Molaro < lupus @ ximian . com >
* InitializationEventAttribute . cs : fixed ExtractValuesFromCell , . net aspx parser not always creates ExtractTemplateValuesMethod delegate
2003 - 02 - 14 Gonzalo Paniagua Javier < gonzalo @ ximian . com >
2008 - 08 - 04 Zoltan Varga < vargaz @ gmail . com >
* TreeView . cs : Use LostFocus instead of Leave for
Merge pull request from cmp - / remove - obsolete - stack - checks - win32
removed test file
[ profiler ] Remove the buffer size hack in ensure_logbuf_unsafe ( ) .
New test .
* ILParser . jay : Create strings from byte arrays . Fix parsing of
2002 - 12 - 19 Dietmar Maurer < dietmar @ ximian . com >
Flush
Add span - based CopyTo and CopyToAsync methods ( dotnet / coreclr )
[ threadpool - ms ] Fix incorrect use of mono_rand .
2002 - 03 - 14 Ville Palo < vi64pa @ koti . soon . fi >
[ coop ] Add a handle frame in construct_culture ( )
2002 - 07 - 30 Gonzalo Paniagua Javier < gonzalo @ ximian . com >
build CloneCopy ( ) test with ReadOnly field
Fri Jul 12 16 : 58 : 52 CEST 2002 Paolo Molaro < lupus @ ximian . com >
2009 - 01 - 09 Miguel de Icaza < miguel @ novell . com >
Improve more test ( )
* RichBoxTest . cs : Disabled ReadOnly test until I can test it on linux .
* NunitWebTest . cs : fixed HtmltoXml , removes spaces and crlf before converting
2008 - 05 - 10 Sebastien Pouliot < sebastien @ ximian . com >
2009 - 01 - 29 Zoltan Varga < vargaz @ gmail . com >
[ jit ] Clean up usage of signature variables .
[ bcl ] Add new files for the g_shell_parse
2010 - 02 - 09 Atsushi Enomoto < atsushi @ ximian . com >
2008 - 10 - 12 Marek Safar < marek . safar @ gmail . com >
2006 - 08 - 15 Martin Baulig < martin @ ximian . com >
2008 - 05 - 14 Rodrigo Kumpera < rkumpera @ novell . com >
- Implemented OnActivated / OnDeactivate Events and Activate ( ) method
* driver . cs : Added GetTypeCode test .
typo fix
[ jit ] Add a new ICALL_ADDR_CALL patch type which is only used on calls , since ICALL_ADDR might be used without a call .
flush
New tests .
2005 - 11 - 04 Sebastien Pouliot < sebastien @ ximian . com >
Updated .
2003 - 05 - 02 Miguel de Icaza < miguel @ ximian . com >
[ jit ] Fix the EqualityComparer . get_Default ( ) optimization for now , it seems to cause failures in the corlib tests on the ios simulator . ( )
[ tests ] Rework how we AOT compile tests + General cleanup of make targets ( )
2004 - 09 - 11 Chris Toshok < toshok @ ximian . com >
* * * empty log message * * *
Add stub for NET_2_0 LockBits
Fix 1 . x build
2010 - 01 - 02 Zoltan Varga < vargaz @ gmail . com >
[ xbuild ] Add ItemDefinitionGroup basic handling
2009 - 08 - 16 Mark Probst < mark . probst @ gmail . com >
Move Interlocked
Merge pull request from IgorTkachev / master
2003 - 10 - 06 Sebastien Pouliot < spouliot @ videotron . ca >
2008 - 10 - 25 Marek Habersack < mhabersack @ novell . com >
2007 - 12 - 13 Zoltan Varga < vargaz @ gmail . com >
1 . 0 . 1
2004 - 09 - 29 Zoltan Varga < vargaz @ freemail . hu >
Revert r84984 and r85314
revert
[ bcl ] Update csproj files
[ ci ] Remove - f option from genhtml during code coverage
[ tests ] Xamarin . Android has the desktop behavior
Fixed nunitlite
2005 - 11 - 12 Gonzalo Paniagua Javier < gonzalo @ ximian . com >
Fixed to JAVAEE interpreter regarding conversion .
[ bcl ] Update BCL Linked Size
Merge pull request from DavidKarlas / cacheThreadInfo
Remove doctools - related test files .
2002 - 02 - 05 Tim Coleman < tim @ timcoleman . com >
[ runtime ] Fix a warning .
[ aot ] Fix a case where an ftnptr was invoked as a method address .
2002 - 03 - 27 Gaurav Vaish < gvaish @ iitk . ac . in >
* mono - api - diff . cs : Fixed line endings .
2004 - 01 - 10 Atsushi Enomoto < atsushi @ ximian . com >
[ System ] Remove duplicate file entry
Remove redundant
merge r108940 - 108941
2007 - 08 - 28 Jb Evain < jbevain @ novell . com >
2002 - 08 - 06 Sebastien Pouliot < spouliot @ videotron . ca >
2004 - 05 - 08 Atsushi Enomoto < atsushi @ ximian . com >
2003 - 04 - 26 Ville Palo < vi64pa @ kolumbus . fi >
[ profiler ] Clarify docs for mono_profiler_set_sample_mode ( ) .
2009 - 01 - 24 Zoltan Varga < vargaz @ gmail . com >
2004 - 06 - 23 Nick Drochak < ndrochak @ ieee . org >
2004 - 09 - 14 Sebastien Pouliot < sebastien @ ximian . com >
[ jit ] Fix the support for MONO_ARCH_NOTIFY_PENDING_EXC , it only works on amd64 , and its not worth the complexity .
2005 - 09 - 15 Sebastien Pouliot < sebastien @ ximian . com >
2006 - 04 - 20 Atsushi Enomoto < atsushi @ ximian . com >
[ threadpool - io ] Remove sparse - array for poll backend
[ ci ] ManualResetEventSlim . SpinCount depends on the number of CPU / cores and we can', "'", 't assume everything as multiple CPU / cores [ ]
2005 - 06 - 15 Peter Bartok < pbartok @ novell . com >
In . :
[ threadpool ] Fix the build .
2010 - 06 - 17 Rodrigo Kumpera < rkumpera @ novell . com >
[ coop ] Add a handle frame in construct_culture ( )
[ sgen ] New protocol entry for processing of individual reference .
* EventLog . cs : Fixed Designer attribute to match MS . NET .
* NunitWebTest . cs : fixed HtmltoXml , removes spaces and crlf before converting
Merge pull request from esdrubal / fileinfo_moveto
partially addressing bug
2005 - 06 - 19 Sebastien Pouliot < sebastien @ ximian . com >
2002 - 05 - 19 Gaurav Vaish < gvaish @ iitk . ac . in >
from head
Revert 67496 as it makes running Paint . NET impossible
[ coop ] Remove try { . . . } finally { . ( )
2002 - 01 - 26 Rodrigo Moya < rodrigo @ ximian . com >
2005 - 03 - 22 Martin Baulig < martin @ ximian . com >
2009 - 01 - 26 Miguel de Icaza < miguel @ novell . com >
eglib : checking for locale_charset function
[ System ] : Cleanup internal TLS APIs in preparation of adding renegotiation support . ( )
2005 - 09 - 24 Zoltan Varga < vargaz @ gmail . com >
2005 - 09 - 04 Alexander Olk < alex . olk @ googlemail . com >
[ genmdesc ] Improved unknown specifier error handling for Perl version .
add >
* man / util / Makefile ( EXTRA_DISTFILES ) : Remove
2006 - 06 - 10 Gonzalo Paniagua Javier < gonzalo @ ximian . com >
2004 - 07 - 11 Ben Maurer < bmaurer @ users . sourceforge . net >
New tests .
[ interp ] introduce inlining private
2004 - 01 - 19 David Sheldon < dave - mono @ earth . li >
2004 - 09 - 01 Ben Maurer < bmaurer @ users . sourceforge . net >
spec fixes
2004 - 01 - 15 Zoltan Varga < vargaz @ freemail . hu >
2005 - 02 - 04 Lluis Sanchez Gual < lluis @ novell . com >
2003 - 03 - 13 Martin Baulig < martin @ ximian . com >
* SortedDictionary . cs ( ICollection . Contains ) : Use EqualityComparer
Update
[ wasm ] [ debugger ] Clear valuetypes cache on Debugger . resume ( )
[ corlib ] Add a workaround for redirected StandardInput writes . Fixes
2010 - 05 - 06 Marek Habersack < mhabersack @ novell . com >
2010 - 07 - 12 Atsushi Enomoto < atsushi @ ximian . com >
2008 - 03 - 13 Zoltan Varga < vargaz @ gmail . com >
Improved test
* DataColumnCollectionTest . cs : Added test for FormatProvider ,
* * * * Merged r37266 from MCS * * * *
Thu Oct 3 16 : 14 : 37 CEST 2003 Paolo Molaro < lupus @ ximian . com >
* * * empty log message * * *
[ System . Net . Http ] Add IVT to Xamarin . * assemblies and update Mac
Tiny fix :
Fix System . ComponentModel . DataErrorsChangedEventArgs API .
ignore FSW test
2005 - 01 - 19 Sebastien Pouliot < sebastien @ ximian . com >
2008 - 10 - 08 Jb Evain < jbevain @ novell . com >
impl . CEQ after lreg , more bug fixes
[ interp ] Track MINT_MONO_LDPTR have constants ( )
Initialization of default members .
use style_attach
Also set readerDoneEvent when exiting an upgradeable lock and reader condition is correct .
[ sgen ] GrayQueueEntry struct instead of just char * .
* CompareInfo . cs : Marked UseManagedCollation internal to have API
[ mcs ] Don', "'", 't hardcode the deprecation notice for minimal or quiet logging
2004 - 07 - 03 Sebastien Pouliot < sebastien @ ximian . com >
2010 - 04 - 06 Marek Safar < marek . safar @ gmail . com >
New test .
[ runtime ] Fix a warning .
002 - 07 - 17 Tim Coleman < tim @ timcoleman . com >
[ arm ] Fix crashes on ios9 by using 16kb trampoline pages .
2006 - 04 - 23 Marek Sieradzki < marek . sieradzki @ gmail . com >
Flush
2010 - 05 - 23 Zoltan Varga < vargaz @ gmail . com >
2006 - 01 - 15 Sebastien Pouliot < sebastien @ ximian . com >
more docu
[ interp ] Remove assertion on ` info ` for socket syscall abort
2006 - 01 - 13 Sebastien Pouliot < sebastien @ ximian . com >
2005 - 02 - 14 Lluis Sanchez Gual < lluis @ ximian . com >
[ System . Private . CoreLib ] Add Nullable runtime support
Fix
2003 - 07 - 09 Andreas Nahr < ClassDevelopment @ A - SoftTech . com >
* DataGrid . cs : Minor fixes .
2004 - 12 - 11 Gonzalo Paniagua Javier < gonzalo @ ximian . com >
Add flags and serializable attributes .
2005 - 10 - 28 Atsushi Enomoto < atsushi @ ximian . com >
2005 - 02 - 10 Gonzalo Paniagua Javier < gonzalo @ ximian . com >
2004 - 09 - 23 Jordi Mas i Hernandez < jordi @ ximian . com >
[ runtime ] Fix the build .
2008 - 01 - 09 Jb Evain < jbevain @ novell . com >
2008 - 02 - 04 Marek Safar < marek . safar @ gmail . com >
2004 - 06 - 25 Atsushi Enomoto < atsushi @ ximian . com >
[ runtime ] Use mono_set_pending_exception ( ) in some places in icalls .
2005 - 09 - 29 Martin Baulig < martin @ ximian . com >
2009 - 08 - 17 Gonzalo Paniagua Javier < gonzalo @ novell . com >
2009 - 09 - 24 Atsushi Enomoto < atsushi @ ximian . com >
* main . c ( dis_method_list ) : Oops . Check method_context for null .
[ runtime ] Fix the build .
Merge pull request from knocte / ntlm_fixes
ups from the last testing .
2009 - 01 - 14 Atsushi Enomoto < atsushi @ ximian . com >
2004 - 11 - 20 Atsushi Enomoto < atsushi @ ximian . com >
2004 - 09 - 11 Nick Drochak < ndrochak @ ieee . org >
2004 - 12 - 02 Zoltan Varga < vargaz @ freemail . hu >
update . csproj .
Added new ( internal ) System . DefaultUriParser to the build .
* * * empty log message * * *
Reworded and Reformatted the Readme
fix typo
* RemotingConfiguration . cs : Avoid adding " id " and " type " as custom
2006 - 08 - 30 Peter Dennis Bartok < pbartok @ novell . com >
[ System . XML ] Fix handling of global variables in XSLs ( )
2009 - 11 - 24 Rodrigo Kumpera < rkumpera @ novell . com >
[ llvmonly ] Add a _checked version for mono_class_from_name ( ) .
[ System . Reflection ] Disable test except on ARM64 ( )
[ mono ] Implement Arm intrinsics : ArmBase , Crc32 ( )
2010 - 04 - 15 Marek Habersack < mhabersack @ novell . com >
[ msvc ] Update csproj files
2005 - 04 - 19 Zoltan Varga < vargaz @ freemail . hu >
2009 - 01 - 24 Miguel de Icaza < miguel @ novell . com >
2005 - 04 - 14 Atsushi Enomoto < atsushi @ ximian . com >
2008 - 09 - 04 Carlos Alberto Cortez < calberto . cortez @ gmail . com >
Lower min / max SSE operations on x64 ( )
2005 - 09 - 08 Martin Baulig < martin @ ximian . com >
2004 - 05 - 11 Sebastien Pouliot < sebastien @ ximian . com >
[ wasm ] [ debugger ] Hash messages and sessions directly ( )
[ mcs ] Cached labels definite assignment check does not need exact match but anything larger than original set satisfies the condition . Fixes
New test .
2010 - 06 - 13 Mark Probst < mark . probst @ gmail . com >
2007 - 04 - 31 Sebastien Pouliot < sebastien @ ximian . com >
Made tests compatible with NUnit 2 . 4
New test .
2005 - 05 - 13 Zoltan Varga < vargaz @ freemail . hu >
[ asp . net ] Do nothing if null SessionStateStoreData is passed to SessionStateServerHandler . SetAndReleaseItemExclusive
Merge pull request from kasthack / sysweb - import
2008 - 07 - 11 Jb Evain < jbevain @ novell . com >
Add LinkDemand and serializable attributes .
2004 - 02 - 13 Atsushi Enomoto < atsushi @ ximian . com >
[ reflection ] Harder the custom attribute code against common type load errors .
2008 - 12 - 04 Marek Safar < marek . safar @ gmail . com >
Up
2009 - 04 - 12 Sebastien Pouliot < sebastien @ ximian . com >
Merge pull request from cmp - / fix - sdb - trampoline - windows - x64
2009 - 08 - 24 Zoltan Varga < vargaz @ gmail . com >
2008 - 01 - 23 Jb Evain < jbevain @ novell . com >
[ runtime ] Fix the build .
Include the build .
2002 - 09 - 10 Daniel Morgan < danmorg @ sc . rr . com >
[ corlib ] Use mono_debug_method_lookup_location ( ) instead of mono_debug_symfile_lookup_location ( ) , the latter doesn', "'", 't support portable pdb files . Use mono_debug_free_source_location ( ) for freeing . Fixes . ( )
[ runtime ] Add gen - compact - seq - points option to MONO_DEBUG .
from head
Add a mono_hwcap_init ( ) call to parse_optimizations ( ) .
Updated test for bug 42135 .
In ilasm / tests :
2004 - 08 - 23 Zoltan Varga < vargaz @ freemail . hu >
System . Drawing . vmwcsproj : Added SystemIcons . jvm . cs , excluded SystemIcons . cs
2005 - 06 - 04 Gonzalo Paniagua Javier < gonzalo @ ximian . com >
[ arm ] Allocate thunk space for tail calls . Fixes .
2005 - 12 - 04 Sureshkumar T < tsureshkumar @ novell . com >
[ threadpool - ms ] Fix incorrect use of mono_rand .
[ jit ] Add a new ICALL_ADDR_CALL patch type which is only used on calls , since ICALL_ADDR might be used without a call .
2008 - 02 - 25 Marek Habersack < mhabersack @ novell . com >
2007 - 09 - 07 Jb Evain < jbevain @ novell . com >
2007 - 04 - 12 Jonathan Pobst < monkey @ jpobst . com >
2010 - 04 - 06 Sebastien Pouliot < sebastien @ ximian . com >
[', ['2008 - layer ] Make _WAPI_PRIVATE_MAX_SLOTS and _WAPI_HANDLE_INITIAL_COUNT definition private
2003 - 02 - 21 Zoltan Varga < vargaz @ freemail . hu >
2008 - 08 - 24 Rodrigo Kumpera < rkumpera @ novell . com >
Bring jbevain / cecil', "'", 's commit 4b68f887b9a6a0a2b5ae268545f8d2d8ae3abe0d , Add support for reading the new target platforms in C # 5 : ARM and AnyCPU32BitPreferred .
Fix the previous marshalling change so it only applies to chars , not structures which contain char fields .
2009 - 01 - 23 Zoltan Varga < vargaz @ gmail . com >
* main . c ( dis_method_list ) : Oops . Check method_context for null .
2006 - 10 - 05 Chris Toshok < toshok @ ximian . com >
[ System ] Add IPv6 support to FtpWebRequest
2009 - 01 - 26 Mark Probst < mark . probst @ gmail . com >
2009 - 05 - 23 Rodrigo Kumpera < rkumpera @ novell . com >
2009 - 09 - 22 Zoltan Varga < vargaz @ gmail . com >
PR : Follow jumps when getting the plt entry for a call .
[ ci ] Try to enable parallel make on RPM build
New tests .
2008 - 01 - 17 Marek Safar < marek . safar @ gmail . com >
[ jit ] Add support for overriding the alignment calculations done by the runtime during AOT . This is needed because the alignment of some types , notable i8 / double can be different between the host and the target platform .
Properly commit :
2006 - 04 - 29 Rolf Bjarne Kvinge < RKvinge @ novell . com >
2005 - 06 - 10 Lluis Sanchez Gual < lluis @ novell . com >
2005 - 09 - 20 Gonzalo Paniagua Javier < gonzalo @ ximian . com >
System . Drawing . dll . sources : Added DbException . cs , SqlClientMetaDataCollectionNames . cs
more docu
typo fix
Remove duplicate doctools .
[ coop ] Remove dead / error code .
2007 - 03 - 02 Martin Baulig < martin @ ximian . com >
2007 - 01 - 26 Marek Sieradzki < marek . sieradzki @ gmail . com >
2010 - 05 - 05 Rodrigo Kumpera < rkumpera @ novell . com >
2009 - 11 - 24 Rodrigo Kumpera < rkumpera @ novell . com >
[ sgen ] Kill SgenThreadInfo : : thread_is_dying in favor of the thread - info state information .
Small fix .
Up
* TreeView . cs : Use LostFocus instead of Leave for
* TreeView . cs : Use LostFocus instead of Leave for
Merge pull request from lateralusX / lateralusX / fix - simd - llvm - issue
2009 - 06 - 09 Zoltan Varga < vargaz @ gmail . com >
New test .
[ gensources ] : Add platform / project logic to tests .
Ensure the condition
[ sgen ] Don', "'", 't wait for sweep finish when allocating major object
Merge pull request from evincarofautumn / c + + - compat
2007 - 04 - 04 Atsushi Enomoto < atsushi @ ximian . com >
2008 - 06 - 25 Marek Habersack < mhabersack @ novell . com >
[ Mono . Security ] : Add API to let MonoTlsProvider provider a custom system certificate validator .
2005 - 12 - 04 Cesar Lopez Nataren < cnataren @ novell . com >
Updated .
2002 - 08 - 16 Dietmar Maurer < dietmar @ ximian . com >
[ sgen ] Fix build on C99 .
2007 - 05 - 16 Atsushi Enomoto < atsushi @ ximian . com >
normalize line endings
* SiteMapProviderCollection . cs : Fixed warning .
2001 - 11 - 21 Dietmar Maurer < dietmar @ ximian . com >
2003 - 12 - 25 Dick Porter < dick @ ximian . com >
[ JIT ] Handle already gsharedvt methods correctly in add_extra_method_with_depth ( ) .
[ runtime ] Remove some dead code .
[ mini ] Add missing membars when initializing rgctx entries ( )
[ Mono . Security ] : Add API to let MonoTlsProvider provider a custom system certificate validator .
2003 - 11 - 28 Patrik Torstensson
New tests .
2007 - 09 - 20 Atsushi Enomoto < atsushi @ ximian . com >
* man / util / Makefile ( EXTRA_DISTFILES ) : Remove
PR feedback : GetElement ( 0 ) - > Sse2 . StoreLow
Fix typo . ( )
2005 - 06 - 14 Sebastien Pouliot < sebastien @ ximian . com >
Add a dummy Facade of a bit
2009 - 06 - 08 Jonathan Chambers < joncham @ gmail . com >
2004 - 07 - 26 Atsushi Enomoto < atsushi @ ximian . com >
* MdiChildContext . cs : When imagelists are used the image area is
[ jit ] Fix the peephole optimization added by 9fc7c1de3a2f7d42c732b20e272d9d6693720012 , compare_imm + cond_exc_imm can only be eliminated if the comparison is false .
* OdbcError . cs , XplatUIX11 . cs , SoapMethodAttribute . cs ,
[ sgen ] Fix a toggleref test .
[ System ] : Make internals visible to Mono . Security .
In class / Microsoft . Build . Tasks / Microsoft . Build . Tasks :
2008 - 05 - 28 Andreas Nahr < ClassDevelopment @ A - SoftTech . com >
* MethodDef . cs : Use new culture info for constructing the
2005 - 09 - 11 Gonzalo Paniagua Javier < gonzalo @ ximian . com >
Set svn : eol - style = native , delete svn : executable .
2005 - 03 - 24 Maverson Eduardo Schulze Rosa < maverson @ gmail . com >
[ jit ] Add support for overriding the alignment calculations done by the runtime during AOT . This is needed because the alignment of some types , notable i8 / double can be different between the host and the target platform .
Disable the test .
[ reflection ] Fix build
* X11Display . cs :
Add to the distribution
removes warning CS0168 .
Use BitOperations in preperation for Jackson', "'", 's jay patch
. cvsignore
[ mcs ] Object initializer for events which can be initialized using field expression
2007 - 01 - 30 Marek Habersack < mhabersack @ novell . com >
2004 - 05 - 13 Atsushi Enomoto < atsushi @ ximian . com >
* driver . cs : Use Mono . Security . StrongNameManager to perform strongname
[ netcore ] Convert netcore / Makefile . am to a normal makefile , get the configure variables from config . make . ( )
eglib : checking for locale_charset function
Add StatusBarDrawItem stuff to build
2010 - 05 - 28 Rodrigo Kumpera < rkumpera @ novell . com >
Slience CVS for a bit
2002 - 10 - 16 Dietmar Maurer < dietmar @ ximian . com >
[ runtime ] Add support for using COM types from corlib .
2005 - 01 - 16 Zoltan Varga < vargaz @ freemail . hu >
2008 - 07 - 28 Ivan N . Zlatev < contact @ i - nz . net >
Merge pull request from ninjarobot / fix - xml - ser - tests
Add TLS offset probing to mach - arm in the same way it', "'", 's done on x86 .
2009 - 05 - 05 Marek Safar < marek . safar @ gmail . com >
New tests .
* EventLog . cs : Added
2003 - 10 - 23 Zoltan Varga < vargaz @ freemail . hu >
adjust
2008 - 08 - 27 Gonzalo Paniagua Javier < gonzalo @ novell . com >
[ io - layer ] Make _WAPI_PRIVATE_MAX_SLOTS and _WAPI_HANDLE_INITIAL_COUNT definition private
2005 - 12 - 11 Martin Baulig < martin @ ximian . com >
2003 - 09 - 08 Zoltan Varga < vargaz @ freemail . hu >
[ Microsoft . Build . BuildEngine ] Remove dead files
Revert " [ mini ] Remove unneeded restriction on fast delegate creation for context - using generic delegate under gshared . ( ) "
2004 - 09 - 07 Atsushi Enomoto < atsushi @ ximian . com >
* InitializationEventAttribute . cs : New .
2003 - 05 - 04 Miguel de Icaza < miguel @ ximian . com >
Another fix for bug interface method .
[ jit ] Simplify the seq point code used for soft breakpoints .
1 . x build fix
2009 - 03 - 11 Bill Holmes < billholmes54 @ gmail . com >
[ System . EnterpriseServices ] Rename method parameters to match . NET contract
[ System . EnterpriseServices ] Rename method parameters to match . NET contract
[ Coop ] ves_icall_RuntimeFieldInfo_GetRawConstantValue . ( )
[ perf ] Turn . c . ( )
2005 - 03 - 04 Atsushi Enomoto < atsushi @ ximian . com >
[ System . Private . CoreLib ] Add Nullable runtime support
2005 - 01 - 06 Martin Baulig < martin @ ximian . com >
[ io - layer ] Make _WAPI_PRIVATE_MAX_SLOTS and _WAPI_HANDLE_INITIAL_COUNT definition private
2009 - 01 - 10 Atsushi Enomoto < atsushi @ ximian . com >
* UnixEnvironment . cs
2008 - 09 - 24 Jonathan Pobst < monkey @ jpobst . com >
[ corlib ] Interning of utc literals in TimeZoneInfo conversion . Fixes
2008 - 10 - 08 Atsushi Enomoto < atsushi @ ximian . com >
Implemented respective AddWithValue methods
2005 - 07 - 22 Cesar Lopez Nataren < cnataren @ novell . com >
2002 - 09 - 02 Tim Coleman < tim @ timcoleman . com >
* ComboBox . cs : Forward ContextMenu to the underlying textbox , if any .
* System . Configuration . dll . sources , System . Configuration20 . vmwcsproj : removed OrdinalComparer . cs
[ profiler ] Emit a debug message to a larger verbosity level .
Fix a warning .
2005 - 04 - 13 Atsushi Enomoto < atsushi @ ximian . com >
TARGET_JVM
Document auto - generated comments .
Fix more verbose
Copied remotely
[ profiler ] Add myself as author .
2008 - 11 - 20 Zoltan Varga < vargaz @ gmail . com >
Small improvement
2006 - 09 - 20 Miguel de Icaza < miguel @ novell . com >
2008 - 06 - 08 Andreas Faerber < andreas . faerber @ web . de >
2004 - 12 - 15 Martin Baulig < martin @ ximian . com >
Fixed NormalizeArgs support
TARGET_JVM
Add stub files
2010 - 07 - 27 Zoltan Varga < vargaz @ gmail . com >
Flush
Merge pull request from alexrp / revert - tp - monitor
* TabControl . cs : Use new culture info method for constructing the
* * * * Merged r37266 from MCS * * * *
[ mini ] Implement Arm intrinsics : ArmBase , Crc32 ( )
Bump API snapshot submodule
* X11Display . cs : Added test for .
2005 - 02 - 22 Zoltan Varga < vargaz @ freemail . hu >
2008 - 06 - 24 Mark Probst < mark . probst @ gmail . com >
this test works on mono
2008 - 03 - 04 Zoltan Varga < vargaz @ gmail . com >
2006 - 09 - 29 Gonzalo Paniagua Javier < gonzalo @ ximian . com >
[ jit ] Precompute a bit more information for delegate trampolines to speed them up .
2005 - 03 - 06 Gonzalo Paniagua Javier < gonzalo @ ximian . com >
Ooops , fix the location check in Error and Warning .

