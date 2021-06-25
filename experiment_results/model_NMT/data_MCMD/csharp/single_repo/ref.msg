Merge pull request from kumpera / cleanup_p1
2010 - 05 - 19 Jb Evain < jbevain @ novell . com >
Merge pull request from directhex / fix - kfreebsd
Nullable : ManualResetEventSlim ( dotnet / coreclr )
2007 - 02 - 04 yonik < yonik @ mainsoft . com >
2005 - 04 - 08 Ankit Jain < radical @ corewars . org >
2004 - 01 - 12 Atsushi Enomoto < atsushi @ ximian . com >
Sigh , more build fixes .
Merge pull request from mono / bockbuild - integration
2006 - 01 - 30 Gonzalo Paniagua Javier < gonzalo @ ximian . com >
2008 - 07 - 24 Rodrigo Kumpera < rkumpera @ novell . com >
Change behavior of HyperLinkField bound field key should be case insensitive .
2008 - 11 - 21 Zoltan Varga < vargaz @ gmail . com >
[ build ] Print CSC when compiling with roslyn
2010 - 02 - 11 Rodrigo Kumpera < rkumpera @ novell . com >
2003 - 02 - 05 Nick Drochak < ndrochak @ gol . com >
Undocumented new targets ' compiler - tests - net_2_0 ' and
Merge pull request from BrzVlad / feature - los - caching
README : Replace - - with - profile2 with - - with - profile4 .
Fix bug 48516
Directory for compiled libs to go into .
2008 - 05 - 07 Robert Jordan < robertj @ gmx . net >
2006 - 09 - 20 Gonzalo Paniagua Javier < gonzalo @ ximian . com >
Bump ikvm module
* class . cs : Added support for Shadows modifier in Variables
cleol
2005 - 12 - 03 Senganal T < tsenganal @ novell . com >
Fix
2004 - 06 - 18 Atsushi Enomoto < atsushi @ ximian . com >
[ sgen ] Add major concurrent collector stw time to " Major GC time " counter
merged the cli and metadata directories - everything is now in metadata
Fix a warning .
Added Click event to menus
2008 - 03 - 24 Zoltan Varga < vargaz @ gmail . com >
update csproj .
2008 - 10 - 10 Rodrigo Kumpera < rkumpera @ novell . com >
2006 - 07 - 07 Atsushi Enomoto < atsushi @ ximian . com >
XamlMember . TypeConverter should return correct converter now .
Ignore mono_use_llvm when JITting in a runtime compiled without LLVM support .
Ignore * . mlpd .
Back out the thread lock changes until they work
* RegexReplace . cs : Fixed line endings .
Fix test - 635 . cs
Adjust OptionalCalendars test so it will work when the linker is enabled
2008 - 06 - 04 George Giolfan < georgegiolfan @ yahoo . com >
2008 - 02 - 19 Jb Evain < jbevain @ novell . com >
2009 - 11 - 12 Zoltan Varga < vargaz @ gmail . com >
[ Socket ] Duplicate logic of AsyncResult . ExecuteWorkItem to SocketAsyncResult
* AsyncMethodResult . cs : Use a ManualResetEvent instead of a mutex ,
Add enum ActiveDirectoryRights
Reverted my last commit ; it was wrong .
* SessionStateServerHandler . cs : Parse connection string . Need to
implement well - known and reserved properties .
2004 - 07 - 02 Zoltan Varga < vargaz @ freemail . hu >
[ mdoc ] Add " internal " ' mdoc x - msitomsx ' command .
Merge pull request from alexanderkyte / finished_mangler
2005 - 09 - 01 Chris Toshok < toshok @ ximian . com >
2004 - 09 - 13 Sebastien Pouliot < sebastien @ ximian . com >
bitmaps is other pixel / bits formats
2006 - 02 - 21 Zoltan Varga < vargaz @ gmail . com >
2008 - 11 - 28 Marek Safar < marek . safar @ gmail . com >
[ binary - reference - assemblies ] Include 4 . 5 Facades in tarball
* inssel . brg : In CEE_ISINST and CEE_CASTCLASS , removed check for proxy .
New function mono_dl_open_runtime_lib that loads runtime libraries .
Add an explanatory comment about ftruncate .
Revert r90735 so the MWF tests build again . Will email the author with the corrections needed .
2005 - 12 - 07 Lluis Sanchez Gual < lluis @ novell . com >
* DerbyProfileProvider . cs : fixed ReadProfileInfo , removed unneeded catch
2003 - 05 - 22 Ben Maurer < bmaurer @ users . sourceforge . net >
[ system . net . http ] Set version to 4 . 0 . 0 . 0 on MOBILE
2005 - 11 - 05 Robert Jordan < robertj @ gmx . net >
per request comment out 3 . 5 stuff
Tue Apr 13 12 : 36 : 29 CEST 2010 Paolo Molaro < lupus @ ximian . com >
* ChannelServices . cs : Configure delayed load channels when a chanel lookup
[ sgen ] Move the logic to determine whenever an ip is in a critical method to sgen code . Avoid loading aot info because its not async safe . ( )
2008 - 04 - 17 Marek Safar < marek . safar @ gmail . com >
2009 - 08 - 07 Atsushi Enomoto < atsushi @ ximian . com >
Merge pull request from BrzVlad / fix - roslyn - ppc
Merge pull request from frebib / fix - strerror_r
Fix aligninig copy and improve overlap detection
[ runtime ] Reorganize mono_class_setup_properties ( ) so most of the work is done outside the loader lock .
2004 - 03 - 10 Zoltan Varga < vargaz @ freemail . hu >
Add explicit interface implementation according to MSDN
2006 - 12 - 05 Rolf Bjarne Kvinge < RKvinge @ novell . com >
Thu Mar 28 16 : 54 : 51 CET 2002 Paolo Molaro < lupus @ ximian . com >
New test .
2004 - 08 - 12 Sebastien Pouliot < sebastien @ ximian . com >
Fixed imm16 range check .
2006 - 02 - 02 Chris Toshok < toshok @ ximian . com >
refactor
[ runtime ] ( remoting . c ) Get rid of mono_error_raise_exception calls
* ReturnMessage . cs : When creating an exception message , copy the call
[ Mono . Security ] : Check ' CertificateValidationHelper . SupportsX509Chain ' before attempting to compute the chain .
2006 - 10 - 27 Dick Porter < dick @ ximian . com >
2005 - 01 - 17 Sebastien Pouliot < sebastien @ ximian . com >
2003 - 07 - 25 Ravi Pratap < ravi @ ximian . com >
2003 - 02 - 08 Zoltan Varga < vargaz @ freemail . hu >
2004 - 02 - 03 Martin Baulig < martin @ ximian . com >
Actually add the file this time : - )
implement such JSF API for being compatible with using of log4j - 1 . 2 . 13 . jar with myfaces .
[ llvm ] Set the MonoJitInfo . from_llvm flag for methods loaded from llvm , it got broken by 1debf3934120547b3003c0ec4ec90bae4b08ee13 . ( )
* xpidl2cs . pl : whitespace clean up
[ llvm ] Remove a wasm limitation which no longer exists . ( )
[ jit ] Fix sgen support in LLVM . Use the new LLVM ' invariant . load ' metadata instead of our ' mono . noalias ' metadata .
Fix an aot crash if field - > type is not initialized .
Add signing of the framework assemblies to the projects using the Open key . ( )
implemented large parts of WCF standard endpoints and configuration support .
2009 - 03 - 10 Zoltan Varga < vargaz @ gmail . com >
Set the X DisplayHandle in System . Drawing
* SerializationTest . cs : Use Assert . AreEqual instead of Assertion . AssertEquals .
Restore RegisteredWaitHandle as we now use the Mono one
compare contextPath ignoring case
2002 - 03 - 12 Duncan Mak < duncan @ ximian . com >
Merge pull request from filipnavara / swf - carbontests
svn path = / trunk / mono / ; revision = 41375
2005 - 08 - 31 Miguel de Icaza < miguel @ novell . com >
Merge pull request from alexrp / master
does
[ System . Configuration ] re - enable standalone tests
Add ( disabled ) unit tests for sync use of CommunicationObject
[ System ] Add some debugging code to BeginConnectMultiple test .
refactor
[ bcl ] Catch exceptions thrown by SerialPortStream . Dispose ( ) when called from the finalizer . Fixes .
more tests
Some more fixes related to displaying correct line number by Sudharsan
2004 - 01 - 05 Gonzalo Paniagua Javier < gonzalo @ ximian . com >
Improved exception stack frame displayed data . Added stack frame method address display .
2008 - 06 - 11 Marek Habersack < mhabersack @ novell . com >
[ sgen ] We used the wrong unmask function in ` sgen - nursery - allocator . c ` .
[ sre ] Wrap mono_image_create_token with HANDLE_FUNCTION_ { ENTER , RETURN }
Tue Jan 24 12 : 34 : 06 CET 2006 Paolo Molaro < lupus @ ximian . com >
2003 - 07 - 23 Dick Porter < dick @ ximian . com >
2009 - 02 - 07 Zoltan Varga < vargaz @ gmail . com >
[ MS . Build ] add MS . Build . dll - specific well - known property , to differentiate from MS . Build . Engine .
2009 - 09 - 26 Mark Probst < mark . probst @ gmail . com >
2002 - 10 - 19 Rachel Hestilow < hestilow @ ximian . com >
ReturnMessage . cs , MethodResponse . cs : OutArgs initialized to an empty array when the the message returns an exception .
Add new CAS tests to the build . Changed Makefile not to warn for 649 while building ( obsolete stuff )
[ genproj ] Added missing git ignored Mono . Cecil . Mdb MSBuild project .
[ runtime ] Convert ValueType icalls to handles .
* TabControl . cs : Fix default tab selection to after TabControl gets focus and not before . Fixes
Fixed incorrect markdown syntax for two links
2006 - 08 - 02 Atsushi Enomoto < atsushi @ ximian . com >
2010 - 06 - 23 Miguel de Icaza < miguel @ novell . com >
* SoapServerFormatterSink . cs : Removed unneded method .
svn path = / branches / mono - 1 - 1 - 9 / mono / ; revision = 51217
Accidentally removed Conversion . cs from CVS . The file will be added back
Rewrite the sgen test driver to make it easy to add configuration and have digestible output .
2007 - 11 - 06 Sebastien Pouliot < sebastien @ ximian . com >
Revert incorrect change .
Workaround Grasshopper ' s bugs with ' Ignore ' attributes .
[ msbuild ] implement CodeTaskFactory i . e . inline tasks .
* binary_serialization_format . htm : renamed filename from having
2005 - 09 - 01 Gonzalo Paniagua Javier < gonzalo @ ximian . com >
2002 - 11 - 10 Ville Palo < vi64pa @ koti . soon . fi >
Fixes build .
Merge pull request from ranma42 / fix - icc - build
* CompressedStack . cs : Added finalizer to match public API of
Add missing test xml files .
[ Mono . Security ] : Make ' MonoTlsSettings . UseServicePointManagerCallback ' a ' bool ? ' .
In Test / System . IO :
Implement Marshal . SetLastWin32Error
2005 - 09 - 06 Atsushi Enomoto < atsushi @ ximian . com >
Fix a JIT crash when - v - v - v - v is used .
Add tutorial for adding more completion types and added new completion
Decorate _image_base__ with __MINGW_LSYMBOL
Tue Sep 27 09 : 32 : 11 EDT 2005 Paolo Molaro < lupus @ ximian . com >
2006 - 08 - 02 Sebastien Pouliot < sebastien @ ximian . com >
2009 - 11 - 08 Rodrigo Kumpera < rkumpera @ novell . com >
2005 - 01 - 31 Ben Maurer < bmaurer @ ximian . com >
Emit sequence point at the beginning of return expression . Fixes
2009 - 12 - 01 Jb Evain < jbevain @ novell . com >
2003 - 12 - 23 Jaroslaw Kowalski < jaak @ zd . com . pl >
[ llvmonly ] Catch all exceptions in mono_jit_exec ( ) to avoid terminating the runtime with an unhelpful libc + + assertion .
( $ ( the_il ) ) : Make resilient to errors .
2005 - 02 - 13 Peter Bartok < pbartok @ novell . com >
[ mono - error ] The reflection code still festered with hidden deps for loader - error .
[ aot ] Don ' t force gsharedvt to be enabled in llvmonly + interp mode . ( )
2008 - 12 - 02 Martin Baulig < martin @ ximian . com >
2004 - 08 - 30 Zoltan Varga < vargaz @ freemail . hu >
2002 - 07 - 31 Duncan Mak < duncan @ ximian . com >
2002 - 05 - 15 Radek Doulik < rodo @ ximian . com >
More generics tests .
Updated SourceLocation with end line and end column
2009 - 07 - 28 Rodrigo Kumpera < rkumpera @ novell . com >
Fixes changelog
updated entry
2002 - 10 - 30 Miguel de Icaza < miguel @ ximian . com >
Add vasprintf to the Cygwin build
[ System . Xml ] Fix casing of System . dll reference in Xslt compiler
2008 - 06 - 01 Zoltan Varga < vargaz @ gmail . com >
2002 - 05 - 23 Dietmar Maurer < dietmar @ ximian . com >
ChangeLog : Updated ChangeLog .
Improve TimeSpan precision ( dotnet / coreclr )
[ msvc ] Fix Debug build by disabling outdated projects in sln
update rsp
2007 - 02 - 05 Marek Safar < marek . safar @ gmail . com >
2008 - 10 - 02 Marek Habersack < mhabersack @ novell . com >
* SoapTypeAttribute . cs : Reverted last change . MS . NET don ' t return the full
Rolling back premature commit .
Reverted previous commit , struct sizes vary way too much between platforms
Override Environment . SpecialFolder . Personal with $ HOME
[ System . Runtime . Caching ] MemoryCacheTest . Trim now checks LRU .
2004 - 05 - 27 Atsushi Enomoto < atsushi @ ximian . com >
2003 - 08 - 06 Miguel de Icaza < miguel @ ximian . com >
Check Condition on task parameter before trying to apply it .
[ runtime ] Remove the last bits of mono_dont_free_global_codeman
2007 - 11 - 23 Marek Habersack < mhabersack @ novell . com >
revert last commit that applied patch Efren
[ amd64 ] More gsharedvt / cleanup work .
2009 - 12 - 03 Zoltan Varga < vargaz @ gmail . com >
Bump corert
2005 - 11 - 19 Zoltan Varga < vargaz @ gmail . com >
2008 - 09 - 22 Marek Safar < marek . safar @ gmail . com >
2006 - 07 - 07 Atsushi Enomoto < atsushi @ ximian . com >
Fixed DoVerticalRendering ( ) and DoHorizontalRendering ( ) .
2004 - 01 - 25 Martin Baulig < martin @ ximian . com >
2009 - 03 - 06 Zoltan Varga < vargaz @ gmail . com >
2009 - 11 - 26 Zoltan Varga < vargaz @ gmail . com >
Fix failing test
2008 - 07 - 28 Jonathan Pobst < monkey @ jpobst . com >
Merge pull request from esdrubal / urilocal
faster Equals
Convert jgac to java_refs , and jre6 to jre
2002 - 10 - 21 Rodrigo Moya < rodrigo @ ximian . com >
2005 - 11 - 04 Sebastien Pouliot < sebastien @ ximian . com >
The real fix
[ msvc ] Remove left over project guid from mono . sln
Added a bulk of tests from Mainsoft repository .
2007 - 11 - 22 Zoltan Varga < vargaz @ gmail . com >
[ sre ] Split out MonoDynamicImage and MonoDynamicStream functions
* TreeView . cs : drag_begin_x and drag_begin_y are now set to - 1 ,
more updates
Fix 4 . 5 framework path used by xbuild
corrected rendering : styles are applied correct
[ mono - threads ] Store handle related data in MonoThreadInfo
[ threads ] Always use the ` suspended ` MonoOSEvent for self - suspend ( )
Add a more reliable way of identifying a Promise to be marshaled .
A new test for bug
Add more details to timing debugging
2009 - 06 - 28 Gonzalo Paniagua Javier < gonzalo @ novell . com >
[ coop ] Don ' t emit safepoints in the wrapper of the polling function .
[ ci ] For checked builds , pass - k to make ( )
[ tools ] Support Booleans in sgen - grep - binprot .
2002 - 09 - 11 Martin Baulig < martin @ gnome . org >
2006 - 03 - 21 Gonzalo Paniagua Javier < gonzalo @ ximian . com >
Adjust indentation
* * * * Merged from MCS * * * *
[ mcs ] Handle try await nested in multiple finally blocks .
2009 - 05 - 15 Carlos Alberto Cortez < calberto . cortez @ gmail . com >
2004 - 05 - 22 Marek Safar < marek . safar @ seznam . cz >
2009 - 03 - 11 Atsushi Enomoto < atsushi @ ximian . com >
* AssemblyBuilderTest . cs : Removed inheritance from deprecated Assertion class ,
[ sgen ] Free lists in blocks must be volatile .
2004 - 01 - 12 Gonzalo Paniagua Javier < gonzalo @ ximian . com >
2004 - 12 - 08 Martin Baulig < martin @ ximian . com >
* CryptoStream . cs : Added [ In , Out ] attributes to Read method .
In System . Collections . Generic :
2009 - 09 - 27 Gonzalo Paniagua Javier < gonzalo @ novell . com >
* Hashtable . cs ( Find ) : Some minor optimizations here .
2005 - 10 - 03 Lluis Sanchez Gual < lluis @ novell . com >
2010 - 05 - 09 Carlos Alberto Cortez < calberto . cortez @ gmail . com >
[ wasm ] Small changes . ( )
Bump NuGet . BuildTasks to upstream dev branch ( )
Merge pull request from kloun / patch - 9
* configure . in : Add required CFLAGS for s390 [ x ] ( - mbackchain )
oops , the change was missing .
2010 - 05 - 06 Rodrigo Kumpera < rkumpera @ novell . com >
Merge pull request from akoeplinger / ci - bump - timeout
* Makefile . am ( MCS_DIRS , MCS_FILES ) : Add back Mono . CSharp . Debugger .
Simplify check for out of range comparison
2009 - 06 - 08 Marek Habersack < mhabersack @ novell . com >
2009 - 12 - 01 Sebastien Pouliot < sebastien @ ximian . com >
2004 - 03 - 12 Marek Safar < marek . safar @ seznam . cz >
2007 - 08 - 15 William Holmes < billholmes54 @ gmail . com >
[ wasm ] make extract_conn_url return null on unexpected data
2006 - 01 - 09 Chris Toshok < toshok @ ximian . com >
[ System ] Replaced while loop logic with an if .
Prevent path fallback for mono - sgen and pedump
* codegen . cs ( EmitContext . DeclSpace ) : Make readonly .
Fixed SWF build on VS . NET .
2006 - 12 - 05 Sebastien Pouliot < sebastien @ ximian . com >
[ profiler ] Increment ref count of an image ' s method while we ' re keeping around pointers to it .
Make use of GetPrefixOfNamespace ( ) in XElement . ToString ( ) . Fixed bug .
2005 - 01 - 21 Atsushi Enomoto < atsushi @ ximian . com >
* Makefile : Added al to net_2_0_SUBDIRS .
2006 - 12 - 01 Chris Toshok < toshok @ ximian . com >
Mark the type owning the entry point
Update to the latest mcs
2002 - 12 - 12 Gonzalo Paniagua Javier < gonzalo @ ximian . com >
[ mono ] Use DefaultDllSearchPaths attribute for PInvokes ( )
[ S . R . Serialization ] fix Json reader member duplicate check index .
use inherited constructor
2009 - 04 - 30 Rodrigo Kumpera < rkumpera @ novell . com >
Merge pull request from echampet / bug - 10001 - 2
2005 - 11 - 15 Zoltan Varga < vargaz @ gmail . com >
( FixedSize ) : Fixed typo .
tiny doc fix for consistency
2004 - 12 - 09 Marek Safar < marek . safar @ seznam . cz >
* TypeStubManager . cs : fixed BindingInfo . ctor , reverted to ONLY_1_1
NaCl build and test scripts
Removed windows only tests .
2004 - 08 - 20 Atsushi Enomoto < atsushi @ ximian . com >
signture fixes
AggregateCacheDependency isn ' t approved yet
2003 - 07 - 31 Ben Maurer < bmaurer @ users . sourceforge . net >
2008 - 05 - 07 Martin Baulig < martin @ ximian . com >
[ docs ] Use Doxygen syntax for Monodoc .
[ wasm ] Add more netcore test suites . ( )
2008 - 03 - 22 Zoltan Varga < vargaz @ gmail . com >
Merge pull request from alexanderkyte / debugger_state
Fix mcs / doc builds
Remove old 1 . 1 conf support in Java build
2005 - 06 - 21 Martin Baulig < martin @ ximian . com >
initialize fontcollection
2005 - 04 - 04 Gonzalo Paniagua Javier < gonzalo @ ximian . com >
2008 - 12 - 16 Marek Safar < marek . safar @ gmail . com >
Merge pull request from spicypixel / hotfix / mono - object - to - string
Bring info on my classes up to date .
[ System . Core ] Allow TryConvert to return null for any reference types ( not only object ) . Fixes
2008 - 01 - 09 Jonathan Pobst < monkey @ jpobst . com >
Fix System . Numerics xunit tests and add them to CI ( )
[ Sys . Data ] fix data adapter tests . ByteFx . Data and Npgsql are gone .
In dis :
2006 - 11 - 10 Alexander Olk < alex . olk @ googlemail . com >
* machine . jvm . config : AspNetDB bug fixes
* * * empty log message * * *
svn path = / trunk / mcs / ; revision = 61511
* ILParser . jay : dashed names aren ' t legal for file names , if the
2003 - 05 - 28 Nick Drochak < ndrochak @ gol . com >
2006 - 02 - 28 Carlos Alberto Cortez < calberto . cortez @ gmail . com >
[ jit ] Use the right type when handling shared versions of RuntimeHelpers . IsReferenceOrContainsReferences ( ) . The previous version only worked by accident . ( )
Merge pull request from esdrubal / concurrentbag
Use new shared code for darwin fast tls .
* System . Drawing / Image . cs : On win32 , when cloning images initialized from
More System . Threading stubs , in the System assembly .
[ metadata ] An array is not an IEnumerator ` 1
2004 - 06 - 09 Umadevi S < sumadevi @ novell . com >
2010 - 02 - 10 Atsushi Enomoto < atsushi @ ximian . com >
2005 - 01 - 10 Lluis Sanchez Gual < lluis @ novell . com >
2008 - 06 - 04 Mark Probst < mark . probst @ gmail . com >
New test .
2005 - 09 - 02 Sebastien Pouliot < sebastien @ ximian . com >
Exceptions during execution cause the block to fault
2010 - 03 - 26 Rodrigo Kumpera < rkumpera @ novell . com >
[ mono ] test_op_il_seq_point test should use a temp file for NUnit results
2003 - 08 - 19 Gonzalo Paniagua Javier < gonzalo @ ximian . com >
Must use an IL version of the test since the underlying bug will be fixed on mcs .
2009 - 05 - 17 Zoltan Varga < vargaz @ gmail . com >
2003 - 12 - 17 Gonzalo Paniagua Javier < gonzalo @ ximian . com >
* ControlStylesTest . cs : Fixed GetStyle for 2 . 0 profile .
2009 - 10 - 02 Atsushi Enomoto < atsushi @ ximian . com >
Bump the size of the epoll threads to 256k on 64 bit systems to avoid problems on ppc64 with 64k page sizes .
socket - io : pass MulticastOption InterfaceIndex into ip_mreqn imr_ifindex
2008 - 01 - 27 Zoltan Varga < vargaz @ gmail . com >
Updated .
bunch of new tests , mostly centering around the lock * attributes
handles AwaitAsync finishing synchronously
[ llvm ] Do a zero extension for bool / char too when loading a volatile variable .
[ jit ] Make it possible to enable gsharedvt in llvmonly mode even if MONO_ARCH_GSHAREDVT_SUPPORTED is not enabled .
2008 - 03 - 17 Jb Evain < jbevain @ novell . com >
[ runtime ] Handle native size of structs with explicit layout and min alignment smaller than default packing . Fixes
[ sgen ] Finish concurrent collection if the heap grows too much
2010 - 02 - 02 Rodrigo B . de Oliveira < rodrigo @ unity3d . com >
[ jit ] Make newarr [ void ] unverified . Fixes .
2009 - 09 - 30 Miguel de Icaza < miguel @ novell . com >
2008 - 07 - 08 Marek Safar < marek . safar @ gmail . com >
[ mono - service ] log service name when reporting ExitCode ! = 0
2006 - 01 - 30 Gonzalo Paniagua Javier < gonzalo @ ximian . com >
* KeyboardLayouts . cs : Eliminate some dead code .
[ corlib ] Fallback to Insertion Sort in the non - generic qsort ( )
2007 - 04 - 03 Alp Toker < alp @ atoker . com >
2004 - 05 - 06 Gonzalo Paniagua Javier < gonzalo @ ximian . com >
2006 - 01 - 12 Alexander Olk < alex . olk @ googlemail . com >
updating mbas text a bit
More informative g_error message
2006 - 05 - 24 Atsushi Enomoto < atsushi @ ximian . com >
[ mono - time ] Replace mono_msec_boottime ( ) with CoreCLR implementation . Fixes https : / / bugzilla . xamarin . com / show_bug . cgi ? id = 58413
2003 - 09 - 29 < cesar @ ciencias . unam . mx >
2005 - 06 - 29 Sebastien Pouliot < sebastien @ ximian . com >
Fix loading from volatile base addresses in the LLVM backend . Fixes .
2006 - 02 - 27 Marek Safar < marek . safar @ seznam . cz >
Cleanup and move nursery allocator tunables to sgen - gc . h
In mcs :
Sniff , sniff
2002 - 07 - 06 Andrew Birkett < adb @ tardis . ed . ac . uk >
Revert temporary hack
[ MonoPosixHelper ] Fix compilation for Android .
[ corlib ] More thread code from reference sources , mostly for argument checks
* roottypes . cs : Rename from tree . cs .
2008 - 01 - 23 Jonathan Pobst < monkey @ jpobst . com >
2003 - 10 - 09 Gonzalo Paniagua Javier < gonzalo @ ximian . com >
[ corlib ] Handle multiple segments in IsolatedStored : : GetFileNames . Fixes bxc
[ mono ] Fix configure to actually disable dllmap on non - mobile platforms ( )
Part of fix - events do not prevent invoker setter unless Handled is true . Also use parent object as eventSender .
[ wasm ] [ xunit ] Re - enable System . MemoryTests and System . SpanTests tests ( )
[ runtime ] Add MONO_EMPTY_SOURCE_FILE to a couple of files . ( )
Don ' t create any . mdb ' s for the PCL Facade Assemblies .
Update .
2003 - 08 - 04 Jerome Laban < jlaban @ wanadoo . fr >
2005 - 04 - 04 Gonzalo Paniagua Javier < gonzalo @ ximian . com >
2006 - 12 - 03 Sebastien Pouliot < sebastien @ ximian . com >
[ System . Numerics ] Cleanup Makefile
Flush
[ corlib ] Add nullable tests
2004 - 05 - 05 Atsushi Enomoto < atsushi @ ximian . com >
Fix a warning .
Bump emscripten to 1 . 39 . 9 . ( )
2004 - 01 - 14 Atsushi Enomoto < atsushi @ ximian . com >
Renamed hash , str to hashtable , string - util
2006 - 12 - 18 Marek Sieradzki < marek . sieradzki @ gmail . com >
2003 - 09 - 16 Gonzalo Paniagua Javier < gonzalo @ ximian . com >
2008 - 03 - 17 Rodrigo Kumpera < rkumpera @ novell . com >
Merge pull request from lambdageek / dev - 42584 - take4
There ' s a tricky namespace - mismatching deserialization in . NET implementation .
2005 - 03 - 17 Miguel de Icaza < miguel @ novell . com >
[ sdks ] Add code to run corlib test suite .
2002 - 06 - 19 Gonzalo Paniagua Javier < gonzalo @ ximian . com >
add myself as an author
* System . Configuration / ConfigurationManagerTest . cs
complete
Bump boringssl submodule to get PPC32 fix
2005 - 02 - 10 Martin Baulig < martin @ ximian . com >
Make this actually work .
[ System . ComponentModel . DataAnnotations ] Replace with . net implementation
2005 - 01 - 10 Gonzalo Paniagua Javier < gonzalo @ ximian . com >
[ interp ] Optimize Span < byte > . Fill ( )
[ jit ] Disable the DYNCALL tests on amd64 again , not all cases are supported right now .
2009 - 10 - 23 Alp Toker < alp @ nuanti . com >
2008 - 03 - 27 Marek Safar < marek . safar @ gmail . com >
2003 - 08 - 19 Gonzalo Paniagua Javier < gonzalo @ ximian . com >
[ docs ] Update formatting in mono - api - string .
added a comment about sigaction
Refactored AddToAny and TakeFromAny to reuse code .
2006 - 06 - 25 Marek Safar < marek . safar @ seznam . cz >
2005 - 02 - 10 Geoff Norton < gnorton @ customerdna . com >
2002 - 04 - 15 Gonzalo Paniagua Javier < gonzalo @ ximian . com >
Merge pull request from lewing / wasm - debug - inspect
ChangeLog : Updated ChangeLog .
typo
[ corlib ] Add FileStream safe handle argument check
2009 - 05 - 26 Zoltan Varga < vargaz @ gmail . com >
* gtest - 260 . cs : New test based on .
Fix support for dynamic methods in collect_method_images ( ) . Fixes .
2008 - 01 - 06 Zoltan Varga < vargaz @ gmail . com >
2008 - 01 - 25 Jb Evain < jbevain @ novell . com >
2010 - 05 - 14 Atsushi Enomoto < atsushi @ ximian . com >
2005 - 02 - 22 Cesar Lopez Nataren < cnataren @ novell . com >
revert
add this
2003 - 02 - 24 Gonzalo Paniagua Javier < gonzalo @ ximian . com >
* Exception . cs : Several changes to make it compatible with MS . NET ( needed
Merge pull request from D - POWER / master
[ runtime ] Add a missing mono_error_init ( ) in a DISABLE_APPDOMAINS codepath .
Implemented fast version of ThreadLocal < T > .
passing those
Update docs
System . Security . Cryptography . X509Certificates enums from . NET Core .
Merge pull request from carlosmn / asp - pc
update2004 - 11 - 19 Marek Safar < marek . safar @ seznam . cz >
[ Socket ] Fix DuplicateAndClose ( )
Warnings cleanup
[ sre ] Make sure a generic definition TypeBuilder has its generic_container setup before we try to instantiate it .
2004 - 02 - 28 Francisco Figueiredo Jr . < fxjrlists @ yahoo . com . br >
flush
[ w32handle ] Const - ify arg of mono_w32handle_namespace_search_handle
mincore ( 2 ) isnot X / Ooen standard
[ io - layer ] Make named handle use their equivalent non - named handle
[ runtime , IO ] Filesystem entries can only be sought for in directories , not files .
[ runtime ] Synthesize IList and IReadOnlyList for the element type of enum errays . Fixes
Some more style nits
* monostyle . 1 : A man page for the monostyle tool .
updates from grammar changes
Add size sample
2006 - 06 - 17 Marek Sieradzki < marek . sieradzki @ gmail . com >
2007 - 04 - 04 Everaldo Canuto < everaldo @ simios . org >
Wed Jan 24 12 : 04 : 39 GMT 2007 Paolo Molaro < lupus @ ximian . com >
2004 - 04 - 09 Zoltan Varga < vargaz @ freemail . hu >
fix failing tests
2008 - 07 - 14 Marek Habersack < mhabersack @ novell . com >
[ monodoc ] Remove . config file when cleaning
changelog
( GetTokenIndex ) : Add implementation for
* MethodCall . cs : Use RemotingServices . GetMethodBaseFromName to get the
2008 - 04 - 01 Ivan N . Zlatev < contact @ i - nz . net >
hm
2007 - 05 - 10 Zoltan Varga < vargaz @ gmail . com >
Updated .
2 more assertions
[ profiler ] Rename proflog . { c , h } to mono - profiler - log . { c , h } for consistency with other profilers .
2008 - 03 - 04 Sebastien Pouliot < sebastien @ ximian . com >
2007 - 08 - 28 Zoltan Varga < vargaz @ gmail . com >
Oooops .
2002 - 03 - 05 Gaurav Vaish < gvaish @ iitk . ac . in >
2008 - 05 - 28 Rodrigo Kumpera < rkumpera @ novell . com >
2008 - 04 - 28 George Giolfan < georgegiolfan @ yahoo . com >
[ mono ] Make bundle filename changes netcore - only ( )
Moved deprecated library
2008 - 06 - 05 Marek Safar < marek . safar @ gmail . com >
* HttpWebRequestTest . cs : Added tests for bug . Added tests
small fix
Add 4 . 0 profile
2008 - 04 - 22 Jonathan Pobst < monkey @ jpobst . com >
2010 - 05 - 14 Atsushi Enomoto < atsushi @ ximian . com >
Contributed under the terms of the MIT / X11 license by
useDefaultCredentials is initialized to false
[ Microsoft . Build ] Fix expected output newline from ProcessWrapper . OutputStreamChanged and ProcessWrapper . ErrorStreamChanged
Remove files that have colons in them to fix SVN for Windows users .
Lldb support code for mono ( )
[ mono ] Add MethodImplOptions . NoOptimization support ( )
* get . c ( get_typedef ) : Return NULL for < Module > type .
[ threads ] Fix crash on unknown policy ( )
XplatUI . cs : Use PlatformID . Unix under NET_2_0 .
fixes integralheight and method ToString
New test .
2006 - 09 - 15 Martin Baulig < martin @ ximian . com >
Merge pull request from michaldobrodenka / master
2008 - 02 - 01 Olivier Dufour < olivier . duff @ gmail . com >
[ corlib ] System . TypeSpec . Parse escaped characters
Update documentation list , sort
Make this actually compile .
2008 - 09 - 27 Miguel de Icaza < miguel @ novell . com >
new test
add another GetLoadedProject ( ) test , this time with ProjectInstance .
2002 - 06 - 23 Rafael Teixeira < rafaelteixeirabr @ hotmail . com >
[ mini ] Document mini_parse_debug_option .
Reimplement StreamWebSocket .
Merge pull request from ermshiperete / 5116
2002 - 10 - 12 Sebastien Pouliot < spouliot @ videotron . ca >
New test
2008 - 03 - 08 Sebastien Pouliot < sebastien @ ximian . com >
* mcs / class / Mono . WebBrowser / README : Warning about dependencies
[ 657623 ] Fix incorrect logic in value type parameter ` is ' check
[ aot ] Avoid emitting a ' methods ' symbol using . set methods , clang ' s integrated assembler doesn ' t seem to support it .
* TypeBuilderTest . cs : Use Assert instead of deprecated Assertion
[ sgen ] Reorganize ` sgen_perform_collection ` for clarity .
Ensure the Monitor . Enter ( ) fastpath doesn ' t apply to Enter ( o , ref taken ) .
2008 - 05 - 19 Marek Habersack < mhabersack @ novell . com >
2004 - 03 - 08 Miguel de Icaza < miguel @ ximian . com >
System . Diagnostics . Process improvements for AIX ( )
2006 - 10 - 13 Chris Toshok < toshok @ ximian . com >
fixed line ending
[ sgen ] Binary protocol entries for scanning objects and vtypes .
2002 - 10 - 19 Tim Coleman ( tim @ timcoleman . com )
Allow GUI front - ends to be used by default .
2003 - 02 - 13 Gonzalo Paniagua Javier < gonzalo @ ximian . com >
Added ServletAuthentication module
Merge pull request from info - lvsys / patch - 1
2005 - 02 - 20 Zoltan Varga < vargaz @ freemail . hu >
2006 - 06 - 14 Sebastien Pouliot < sebastien @ ximian . com >
2009 - 08 - 07 Rodrigo Kumpera < rkumpera @ novell . com >
[ io - layer ] Remove mutex and namedmutex pid field
2009 - 10 - 06 Atsushi Enomoto < atsushi @ ximian . com >
[ jit ] Add a missing call to mini_native_type_replace_type in mini_get_basic_type_from_generic .
* Makefile . am : add test - inline - call - stack - library . cs and test - inline - call - stack . cs
2003 - 02 - 21 Tim Coleman < tim @ timcoleman . com >
This is an untested fix for a bug .
Removed the non - existant test - 340 . cs .
[ System . Net . Http ] Comment out dead code . ( )
[ runtime ] Add fastpaths to the Monitor . Enter ( ) methods .
Deprecate
Add csc - dim script to . gitignore
2003 - 06 - 06 Gonzalo Paniagua Javier < gonzalo @ ximian . com >
Stubbed out System . Data . ObjectSpaces . Query
2004 - 06 - 18 Atsushi Enomoto < atsushi @ ximian . com >
Merge pull request from alexanderkyte / merp_thread_hash
Ignore TestResult . xml too
[ asp . net ] Added a reminder comment about stack scanning in CacheItemPriorityQueue . ResizeHeap
* ILParser . jay : Implement data items
Fix a typo
Make some changes to the stack overflow handling code so it works to some degree even if the kernel doesn ' t give us the fault address .
2010 - 03 - 11 Atsushi Enomoto < atsushi @ ximian . com >
truncated filenames
New test .
Nullable : Enums , Structures from Reflection folder ( dotnet / coreclr )
[ jit ] Print out methods which failed to JIT when using - - regression .
Set ProhibitDtd = false on uri - based Load ( ) . Fixed bug .
Duplicate less mscorlib code in System . Core
Add support for skipping over finally blocks at runtime . First
* PageParser . cs : Add Trace and Trace mode attributes .
Typo fix
TARGET_JVM - NativeObject property
Brush . jvm . cs : added brush transform field
[ acceptance - tests ] Remove folders from Roslyn script that we now ship after 8f130a6e17c372
2003 - 08 - 06 Andreas Nahr < ClassDevelopment @ A - SoftTech . com >
[ mobile_static ] Blacklist runtime tests for mobile_static profile
Remove an unneeded inner ` # ifndef DISABLE_COM ` .
2002 - 05 - 05 Daniel Morgan < danmorg @ sc . rr . com >
2004 - 04 - 28 Sebastien Pouliot < sebastien @ ximian . com >
MIPS : Remove unused variables in mono_arch_find_jit_info
2004 / 08 / 01 Rafael Teixeira < rafaelteixeirabr @ hotmail . com >
2009 - 04 - 30 Zoltan Varga < vargaz @ gmail . com >
2007 - 12 - 27 Jonathan Pobst < monkey @ jpobst . com >
[ runtime ] Remove locking from the mono_get_method_constrained_ . . . ( ) functions , its not needed .
New test .
2007 - 11 - 30 Zoltan Varga < vargaz @ gmail . com >
2005 - 09 - 25 Marek Sieradzki < marek . sieradzki @ gmail . com >
2005 - 06 - 09 Zoltan Varga < vargaz @ freemail . hu >
[ runtime ] Move unit tests to separate directory .
* MainMenu . cs , MenuAPI . cs : Implement Collapse event for MainMenu ( 2 . 0 ) .
[ debugger ] Crash when debugging iOS application that throws ( )
Add mono_gc_thread_detach callback . ( )
2009 - 02 - 23 Marek Safar < marek . safar @ gmail . com >
Update generic . cs
Use SIGXFSZ on FreeBSD too by checking for the __FreeBSD__ define . Like Apple and OpenBSD , it doesn ' t have SIGPWR .
* Image . cs : Save should use encoder . FormatID , not Clsid
Fri Jul 24 17 : 28 : 37 CEST 2009 Steven Munroe < munroesj @ us . ibm . com >
Use ' mcs ' for . NET 4 . 5 .
2006 - 01 - 30 Atsushi Enomoto < atsushi @ ximian . com >
Fix and simplify float conversions
2002 - 09 - 14 Martin Baulig < martin @ gnome . org >
update the ancient README
2004 - 09 - 30 Anirban Bhattacharjee < banirban @ novell . com >
* * * empty log message * * *
2008 - 04 - 28 Jonathan Pobst < monkey @ jpobst . com >
[ System . ComponentModel . Composition ] Remove ReadOnlyDictionary . It ' s not used in 4 . 5 profile ( it implements different interfaces compare to public ReadOnlyDictionary anyway )
2006 - 11 - 18 Jelle Hissink < Jelle . Hissink @ C - it . nl >
These tests run fine but were excluded from the suite . So we fix that ; - )
[ runtime ] Use MonoError for RESOLVE_TYPE
[ ci ] Fix OSX package builds to only be timer - triggered on master
* X11keyboard . cs : Fix last patch , maxval must be less not greater than
[ llvm ] Fix support for LLVM_API_VERSION = = 391 which is used by the release_39 branch in mono ' s llvm fork . ( )
2008 - 07 - 14 Marek Safar < marek . safar @ gmail . com >
* WebConfigurationManager . cs : fixed OpenWebConfiguration , should not lock when checking if configuration is already open
2003 - 12 - 10 Zoltan Varga < vargaz @ freemail . hu >
2006 - 04 - 26 Zoltan Varga < vargaz @ gmail . com >
[ logging ] Default to stdout if MONO_LOG_DEST is unset ( )
Disable a flaky test
add new library for windows
Merge pull request from AxlPr / master
Merge pull request from ludovic - henry / fix - coop - hang
2002 - 03 - 04 Gaurav Vaish < gvaish @ iitk . ac . in >
Checkpoint
Added uninstall target
[ profiler ] Include vtune support for jitted methods in the build system .
2002 - 02 - 06 Gaurav Vaish < gvaish @ iitk . ac . in >
[ profiler ] Fix a warning .
[ sgen ] Use separate object ops with the concurrent collector when there is no evacuation
Add test
New test .
convert value only if it ' s convertable
2004 - 04 - 16 Zoltan Varga < vargaz @ freemail . hu >
3 . 5 version of Sys . Web . Ext is gone
2003 - 05 - 19 Dietmar Maurer < dietmar @ ximian . com >
2001 - 10 - 17 Ravi Pratap < ravi @ ximian . com >
Bump roslyn - binaries submodule
[ bcl ] Fix build dependencies for the corlib vts tests .
[ jit ] Emit div checks if MONO_ARCH_NEED_DIV_CHECK is enabled even when div instructions are not emulated . Enable div checks for tvos .
Commit 29810a78e5b93d8da9fb921d096226d249fc75a5 added unconditional dependency on GetCpuUtilization ( )
[ aot ] Add clangxx option to - - aot ( )
Use common consts in [ InternalsVisibleTo ] PublicKey specifications .
Run more runtime tests on wrench .
Tue Jan 23 18 : 09 : 21 CET 2007 Paolo Molaro < lupus @ ximian . com >
Update
Fix quoted values parsing for custom tostring formats . Fixes
2005 - 11 - 05 Robert Jordan < robertj @ gmx . net >
2006 - 11 - 23 Sebastien Pouliot < sebastien @ ximian . com >
2004 - 06 - 30 Zoltan Varga < vargaz @ freemail . hu >
2008 - 11 - 24 Mark Probst < mark . probst @ gmail . com >
Prepping for monodoc move from monodoc / engine to mcs / class / monodoc . . .
2002 - 02 - 18 Duncan Mak < duncan @ ximian . com >
2002 - 02 - 25 Duncan Mak < duncan @ ximian . com >
New test .
2010 - 07 - 23 Marek Habersack < mhabersack @ novell . com >
2006 - 07 - 13 Gonzalo Paniagua Javier < gonzalo @ ximian . com >
System :
2009 - 11 - 12 Zoltan Varga < vargaz @ gmail . com >
2008 - 09 - 03 Miguel de Icaza < miguel @ novell . com >
2003 - 02 - 14 Dietmar Maurer < dietmar @ ximian . com >
2005 - 01 - 10 Sebastien Pouliot < sebastien @ ximian . com >
* ArrayWithOffset . cs : fixed GetOffset to return int instead of
2004 - 08 - 30 Sebastien Pouliot < sebastien @ ximian . com >
* TraceHandler . cs : Obey localOnly attribute .
Move files which were split out from System . Web into System . Web . ApplicationServices into the latter ' s directory . They no longer need to be in the former , since we no longer build the 2 . 0 profile they were originally part of . This fixes " no rule to make target " errors on tarball builds , caused by those files no longer being shipped in tarballs ( make dist does not traverse a . . / boundary , so it wasn ' t enough for them to be listed in System . Web . ApplicationServices ' manifest by System . Web ' s folder )
Extend HttpClient to use infrastructure in Xamarin . Mac ( via reflection ) to pick right type .
[ runtime ] Install profiler - events . h as a public header .
[ arm ] Fallback to aeabi_read_tp on linux fast tls
* MenuItem . cs : Invalidate menu item rectangle after change Enable
fixed valuetypes marshaling in delegates
Update this
Implement TypeDescriptionProviderService .
2004 - 03 - 23 Sebastien Pouliot < sebastien @ ximian . com >
Mon Nov 18 17 : 52 : 56 CET 2002 Paolo Molaro < lupus @ ximian . com >
Fix some configure tests so they work with clang .
Ooooops .
2006 - 10 - 18 Atsushi Enomotot < atsushi @ ximian . com >
( IfElseStateMachine . State , IfElseStateMachine . Token ) : Remove [ Flags ] attribute .
Remove a comment MD added about a file that wasn ' t committed .
2002 - 12 - 15 Sebastien Pouliot < spouliot @ videotron . ca >
2004 - 02 - 23 Martin Baulig < martin @ ximian . com >
2006 - 04 - 19 Massimiliano Mantione < massi @ ximian . com >
[ System . Core ] Remove generic indirection in ParallelExecuter . Process
[ corlib ] Cope with the reference sources ' inability to support sub - minute DST offsets .
Fixed invalid cast exception
[ tests ] Regression test for
Add a list of historic Mono branches , before we delete them from the server
* PrintingServicesWin32 . cs : Add enumeration flag to get remote
* ExternTable . cs : Revert back to resolving value and class types
Comparing pid instead of handles in GetModules_internal ( )
* System . Design . dll . sources : added ISelectionUIHandler
2008 - 09 - 19 Atsushi Enomoto < atsushi @ ximian . com >
[ asp . net , routing ] Fix for bug . Matching engine must ignore trailing slash in the URLs .
[ runtime ] Use MonoError for mono_object_new
Improve test to check for toplevel namespaced delegates
[ aot ] Open AOT profile files using " rb " so it works on windows .
add to the directory parser all the assemblies found by theme file parsers
add test for static properties
2010 - 05 - 27 Zoltan Varga < vargaz @ gmail . com >
* TreeView . cs : We must use the comparer supplied by TreeViewNodeSorter
2004 - 09 - 30 Zoltan Varga < vargaz @ freemail . hu >
Add to the build
* src / glib . h : Rebase g_return_if_fail ( ) , g_return_val_if_fail ( ) in
2002 - 10 - 16 Miguel de Icaza < miguel @ ximian . com >
Fix the mobile build
2001 - 12 - 22 Gaurav Vaish < gvaish @ iitk . ac . in >
2006 - 01 - 03 Zoltan Varga < vargaz @ gmail . com >
* Control . cs ( OnInvalidated ) : fix NRE relating to last change .
2005 - 02 - 28 Atsushi Enomoto < atsushi @ ximian . com >
Fix reference
* build - csproj2k5 : Add Mono . Mozilla as a reference .
* Control . cs ( ChildNeedsRecreating ) : this should return true if
2004 - 01 - 14 Atsushi Enomoto < atsushi @ ximian . com >
svn path = / trunk / mcs / ; revision = 60910
Make UnloadAppDomain not supported only under TARGET_JVM ( bug 10640 )
2004 - 11 - 22 Marek Safar < marek . safar @ seznam . cz >
2009 - 04 - 03 Zoltan Varga < vargaz @ gmail . com >
2001 - 12 - 04 Ravi Pratap < ravi @ ximian . com >
2005 - 09 - 01 Atsushi Enomoto < atsushi @ ximian . com >
* SettingsPropertyValue . cs : fixed PropertyValue property when it should be deserialized
2008 - 03 - 24 Jonathan Pobst < monkey @ jpobst . com >
add QueryStringConverter String < - > Enum
2006 - 01 - 20 Atsushi Enomoto < atsushi @ ximian . com >
2003 - 01 - 08 Gonzalo Paniagua Javier < gonzalo @ ximian . com >
update todo
2008 - 05 - 30 Zoltan Varga < vargaz @ gmail . com >
2004 - 01 - 09 Atsushi Enomoto < atsushi @ ximian . com >
Fixed MSVC warning about unreferenced local variable .
Novell - 710512 : Fix GetUtcOffset when called with UTC time
2009 - 01 - 07 Zoltan Varga < vargaz @ gmail . com >
fixed RegisterStartupScript
2006 - 11 - 14 Atsushi Enomoto < atsushi @ ximian . com >
Added public property to get the methods .
2008 - 10 - 22 Marek Habersack < mhabersack @ novell . com >
2008 - 07 - 02 Zoltan Varga < vargaz @ gmail . com >
2008 - 04 - 05 Ivan N . Zlatev < contact @ i - nz . net >
2002 - 05 - 10 Dietmar Maurer < dietmar @ ximian . com >
2002 - 08 - 06 Rafael Teixeira < rafaelteixeirabr @ hotmail . com >
Merge pull request from knocte / sock_improvements
Remove deprecated - mno - cygwin flag .
Updated this a bit .
* LinkLabelTest . cs : Put LinkTest and LinkCollectionTest classes outside of
Put back obsolete ` X509Helper2 ` code that is still in use by the legacy implementation .
- Destroys the caret if a window that ' s being destroyed contains it
* mono / jit / x86 . brg : impl . more opcodes
Add helper
* GridViewTest . cs : Code formatting .
[ runtime ] Make merp eventType icall argument
Omit handle type - safety on untested PowerPC , S390X , SPARC , MIPS , etc . ( )
2005 - 09 - 12 Zoltan Varga < vargaz @ gmail . com >
Fix mono - cq under sgen .
2003 - 04 - 26 Pedro Mart � nez Juli � < yoros @ wanadoo . es >
[ WinForms ] Display caret while click on ComboBox ( )
2007 - 02 - 07 Atsushi Enomoto < atsushi @ ximian . com >
2003 - 09 - 01 Sebastien Pouliot < spouliot @ videotron . ca >
Merge pull request from luhenry / threads - rework - shutdown
[ tests ] Remove # to enable tests
[ System . Core / Android ] Update TimeZone database self test .
[ ci ] Fix wrong redirect which would overwrite Make . config
* LinkLabel . cs ( Select ) : if we chain up to the parent , set
Use assembly name and file version constants .
2007 - 06 - 01 Atsushi Enomoto < atsushi @ ximian . com >
Added tests for SizeConverter . cs and PointConverter . cs .
2010 - 02 - 05 Miguel de Icaza < miguel @ novell . com >
2008 - 11 - 12 Marek Safar < marek . safar @ gmail . com >
2005 - 05 - 02 Peter Bartok < pbartok @ novell . com >
[ SWF ] Fix behavior when scrollbar values change in TreeView
[ xBuild ] Add missing VisualStudioVersion propety which was introduced in VS2012
Moving from monodoc / engine to mcs / docs . . .
[ aot ] Remove debugging print .
Bump mono / corefx @ 17dd540 ( )
2002 - 01 - 10 Aaron Weber < aaron @ ximian . com >
add an example I found on the web
oops
2008 - 10 - 11 Miguel de Icaza < miguel @ novell . com >
* ILParser . jay : When modifying extern types , clone them and modify
Adding missing attribute
AOT not supported on Windows , warn the user
Removed debugging code
fixes bug 66533
* emit - x86 . c : ( arch_create_native_wrapper ) : free allocated strings , bug fixes
2008 - 01 - 07 Jb Evain < jbevain @ novell . com >
Fix
2003 - 06 - 15 Sebastien Pouliot < spouliot @ motus . com >
* Control . cs : Set WS_ styles for borders , so that the driver does
MemoryStream implementation from Marcin
2007 - 01 - 23 Dick Porter < dick @ ximian . com >
2006 - 10 - 12 Jonathan Pobst < monkey @ jpobst . com >
2003 - 12 - 19 Dick Porter < dick @ ximian . com >
[ boehm ] Put * _freelists into thread_local_freelists ( as in BDWGC v7 )
2004 - 05 - 07 Sebastien Pouliot < sebastien @ ximian . com >
Clean event registration when wait is unsuccessful
[ utils / hp ] Export mono_thread_hazardous_try_free ( ) for use in the profiler .
2009 - 01 - 19 Zoltan Varga < vargaz @ gmail . com >
2005 - 07 - 01 Martin Baulig < martin @ ximian . com >
[ Winforms ] Calculate GroupBox preferred size the same way as Panel ( )
2009 - 01 - 04 Rodrigo Kumpera < rkumpera @ novell . com >
2008 - 11 - 19 Zoltan Varga < vargaz @ gmail . com >
Tue Jul 22 15 : 48 : 50 CEST 2003 Paolo Molaro < lupus @ ximian . com >
[ runtime ] Perform the reference assembly attribute check using the low level metadata apis outside of a lock .
2006 - 05 - 03 Jb Evain < jbevain @ gmail . com >
2006 - 11 - 08 Sebastien Pouliot < sebastien @ ximian . com >
2006 - 05 - 18 Peter Dennis Bartok < pbartok @ novell . com >
[ 678726 ] Dynamic inside anonoymous methods does not always mutate calling types
2001 - 07 - 26 Miguel de Icaza < miguel @ ximian . com >
Add documentation regarding MONO_TRACE .
Add stubs for some NET 4 . 5 ThreadLocal ctors / methods .
Add . NET 4 . 5 async methods to XmlReader .
2004 - 07 - 08 Zoltan Varga < vargaz @ freemail . hu >
Revert " Disable bug - 18026 . exe when running tests in the CI environment "
Merge pull request from jack - pappas / patch - 4
2002 - 7 - 3 DennisHayes < dennish @ raytek . com >
2003 - 04 - 05 Nick Drochak < ndrochak @ gol . com >
Modify my image entry .
Fix build of mono - basic on ppc64
2003 - 09 - 29 Zoltan Varga < vargaz @ freemail . hu >
2007 - 06 - 12 Jonathan Pobst < monkey @ jpobst . com >
[ reflection ] Set pending loader exception in mono_reflection_get_custom_attrs_data
- Removed several hacks that are no longer needed with the latest
2007 - 09 - 11 Jonathan Pobst < monkey @ jpobst . com >
2001 - 12 - 04 Ravi Pratap < ravi @ ximian . com >
2009 - 02 - 02 Atsushi Enomoto < atsushi @ ximian . com >
[ socket ] Enable Shutdown on socket disposal only on specific platforms
Map OdbcType . Time appropriately . Fixes BNC .
[ mcs ] Inflate referenced current types for imported type definitions . Fixes
[ System ] Use default parsing rules for Uri . IsWellFormedUriString . Fixes
2004 - 09 - 21 Cesar Lopez Nataren < cesar @ ciencias . unam . mx >
Switch sgen to use utils / mono - logger - internal . h for logging .
[ msvc ] Update csproj files ( )
[ loader ] Free some leaked strings in NativeLibrary resolution ( )
a minor code optimization
Thu Dec 1 12 : 59 : 40 EST 2005 Paolo Molaro < lupus @ ximian . com >
[ sgen - bridge ] Remove single - reference non - bridge objects from graph .
[ build ] Add new profile
2008 - 06 - 11 Atsushi Enomoto < atsushi @ ximian . com >
2010 - 07 - 05 Jb Evain < jbevain @ novell . com >
2006 - 01 - 17 Senganal T < tsenganal @ novell . com >
[ runtime ] Actually clean up context - static data segments .
2004 - 09 - 05 Ben Maurer < bmaurer @ users . sourceforge . net >
* Menu . cs : performance refactoring , used AddAttribute overload without encoding on known attribute values , optimized DynamicMenu rendering
Fri Jan 23 21 : 07 : 02 CET 2004 Paolo Molaro < lupus @ ximian . com >
2009 - 04 - 23 Jb Evain < jbevain @ novell . com >
Tue May 21 17 : 25 : 49 CEST 2002 Paolo Molaro < lupus @ ximian . com >
2009 - 01 - 09 Ivan N . Zlatev < contact @ i - nz . net >
Merge pull request from esdrubal / uriserializationinfo
2003 - 12 - 02 Gonzalo Paniagua Javier < gonzalo @ ximian . com >
Fix tests / bug79956 . 2 . il
Implement ServiceEndpoint . EndpointBehaviors
Update copyrights
2006 - 10 - 22 Zoltan Varga < vargaz @ gmail . com >
Updated version of the patch .
[ PerfCounters ] Added " Page File Bytes " which shows VmSwap and made few MacOS counters more accurate
2006 - 07 - 30 Carlos Alberto Cortez < calberto . cortez @ gmail . com >
2005 - 08 - 31 Sebastien Pouliot < sebastien @ ximian . com >
Workaround for building on cygwin
2003 - 12 - 14 Sebastien Pouliot < spouliot @ videotron . ca >
Bump boostrap mono dependency
2008 - 11 - 20 Rodrigo Kumpera < rkumpera @ novell . com >
unicode support ( using iconv )
[ runtime ] Print wrapper methods in the same format as normal methods in stack traces . Fixes .
2009 - 08 - 30 Jb Evain < jbevain @ novell . com >
2002 - 01 - 21 Miguel de Icaza < miguel @ ximian . com >
2010 - 01 - 03 Zoltan Varga < vargaz @ gmail . com >
Recalculate tab rows when resizing
Fix go - mono . com ' s IP address in DnsTest . cs .
[ MonoDroid ] Fix the libmono build .
Add Chris to the maintainers list
added better GetHashCode ( ) implementation
reduce memory use on Attribute ( ) caused by Attributes ( ) .
revert 144379
class should be sealed
Merge pull request from mfoliveira / ppc64el - build - errors - v2 - mit
lifted relational tests
2008 - 11 - 18 Gonzalo Paniagua Javier < gonzalo @ novell . com >
In System . Drawing :
[ interp ] Remove unnecessary recomputation ( )
* ServletWorkerRequest . jvm . cs : fixed ctor , adding welcome page to aspnetconfig folder
2005 - 06 - 24 Marek Safar < marek . safar @ seznam . cz >
Store the domain id of weak references .
[ aot ] Fix a regression introduced by cc946ea5b9e3cbb020c66d8986ecd220ca7ba58c .
2009 - 11 - 30 Marek Habersack < mhabersack @ novell . com >
2009 - 01 - 22 Atsushi Enomoto < atsushi @ ximian . com >
[ csproj ] Update project files
[ runtime ] Generate the proper exit code for block guard trampolines .
ChangeLog .
2002 - 05 - 02 Ravi Pratap < ravi @ ximian . com >
Reenable a sgen test .
fix typo
Wed Apr 7 11 : 28 : 42 CEST 2010 Paolo Molaro < lupus @ ximian . com >
2008 - 07 - 29 Marek Safar < marek . safar @ gmail . com >
* ILParser . jay : Add implemented interfaces
0 . 4
Merge pull request from marek - safar / cecil - update
[ Socket ] Minimize allocations of delegates
Add support for an output path in mono - api - diff ( )
[ arm ] if mtriple is provided , do not set - march ( )
Grasshopper project system now uses csproj extension
2004 - 12 - 10 Marek Safar < marek . safar @ seznam . cz >
2002 - 08 - 17 Jason Diamond < jason @ injektilo . org >
2001 - 09 - 26 Miguel de Icaza < miguel @ ximian . com >
2005 - 04 - 30 Ben Maurer < bmaurer @ ximian . com >
Thu Jan 18 17 : 55 : 22 CET 2007 Paolo Molaro < lupus @ ximian . com >
[ windows ] Prepare VS build properties for proper DLL export / import definitions .
speedup menus drawing
2010 - 02 - 20 Geoff Norton < gnorton @ novell . com >
fixed test suite
HierarchicalDataSourceView . cs : Reverting my earlier patch . Lots of work needs to be done before i can push my DataSourceView changes .
[ bcl ] Remove ONLY_1_1 defines from class libs
[ Sys . Xml ] give understandable message and arguments for Xml_WrongToken .
* RealProxy . cs , RemotingProxy . cs : move the LogicalCallContext
Exclude the new System . IO . Directory unit test for iOS ( no Mono . Unix )
Place break inside if block
2006 - 01 - 02 John Luke < john . luke @ gmail . com >
ChangeLog update .
2005 - 02 - 10 Marek Safar < marek . safar @ seznam . cz >
[ mdoc ] Add some diagnostics
New tests .
2002 - 07 - 25 Tim Coleman < tim @ timcoleman . com >
2005 - 06 - 13 Martin Baulig < martin @ ximian . com >
2005 - 03 - 01 Zoltan Varga < vargaz @ freemail . hu >
2003 - 07 - 28 Sebastien Pouliot < spouliot @ videotron . ca >
2005 - 12 - 09 Atsushi Enomoto < atsushi @ ximian . com >
2006 - 06 - 20 Jb Evain < jbevain @ gmail . com >
Make Mono relocatable
2002 - 10 - 23 Nick Drochak < ndrochak @ gol . com >
2007 - 10 - 30 Atsushi Enomoto < atsushi @ ximian . com >
2007 - 08 - 17 Jonathan Pobst < monkey @ jpobst . com >
2002 - 06 - 20 Ajay kumar Dwivedi < adwiv @ yahoo . com >
2002 - 05 - 20 Miguel de Icaza < miguel @ ximian . com >
Clear entries referencing the domain in reference_queue_clear_for_domain ( ) even if the object is already null . Fixes .
2008 - 07 - 29 Martin Baulig < martin @ ximian . com >
2007 - 09 - 16 Rodrigo Kumpera < rkumpera @ novell . com >
2009 - 07 - 14 Atsushi Enomoto < atsushi @ ximian . com >
[ runtime ] Export mono_personality ( ) , which is accessed from llvm compiled code .
Add rowId to DataRow . Initiate new rows with id - 1 . These is mostly for preformance improvement .
2005 - 06 - 10 Cesar Lopez Nataren < cnataren @ novell . com >
2010 - 01 - 15 Astushi Enomoto < atsushi @ ximian . com >
* src / gptrarray . c : implement g_ptr_array_remove_index_fast
make eglib work with c + +
Modified the Linq Sort ( ) to re - use the Array . Sort ( )
2009 - 06 - 05 Rodrigo Kumpera < rkumpera @ novell . com >
2009 - 12 - 17 Marek Habersack < mhabersack @ novell . com >
2008 - 07 - 17 Ivan N . Zlatev < contact @ i - nz . net >
new source files
2006 - 06 - 06 Mike Kestner < mkestner @ novell . com >
Create Task continuation with less intermediate delegates
* Application . cs : when entering a runloop with a modal , make sure
[ aot ] Fix thumb support with separate llvm modules .
2003 - 10 - 13 Zoltan Varga < vargaz @ freemail . hu >
Inspired by a patch from Alexander Concha , do not crash if we do not find the resource
Set null object for default timer
[ runtime ] Fix the windows build .
2009 - 10 - 21 Marek Safar < marek . safar @ gmail . com >
* include / gc . h : ( GCEventType ) : Added start - stop the world events .
2006 - 02 - 12 Cesar Lopez Nataren < cnataren @ novell . com >
svn path = / trunk / mono / ; revision = 53658
2008 - 04 - 15 Marek Safar < marek . safar @ gmail . com >
[ mini ] Initial tiered compilation work ( )
2010 - 04 - 27 Mark Probst < mark . probst @ gmail . com >
2010 - 04 - 10 Veerapuram Varadhan < vvaradhan @ novell . com >
2002 / 10 / 28 Nick Drochak < ndrochak @ gol . com >
2004 - 12 - 10 Lluis Sanchez Gual < lluis @ novell . com >
New test .
fix typo in the readme . Thanks @ Suchiman
Spell check
2008 - 08 - 29 Marek Safar < marek . safar @ gmail . com >
Mon Jan 29 11 : 30 : 46 CET 2007 Paolo Molaro < lupus @ ximian . com >
Update copyrights
2004 - 12 - 17 Martin Baulig < martin @ ximian . com >
2006 - 11 - 09 Rolf Bjarne Kvinge < RKvinge @ novell . com >
2209 - 01 - 21 Carlos Alberto Cortez < calberto . cortez @ gmail . com >
2009 - 07 - 22 Rodrigo Kumpera < rkumpera @ novell . com >
* FormView . cs : save DefaultMode and DataKeyMode in control state .
Added Changelog
2006 - 05 - 24 Sebastien Pouliot < sebastien @ ximian . com >
updates
Don ' t set NewSlot attribute when base class already implements the interface method and same method exists in derived class
[ netcore ] Add back pushing to blob feed in AzDO pipeline
Sigh , back out
Fix make dist .
2004 - 05 - 17 Ben Maurer < bmaurer @ users . sourceforge . net >
Close AnnouncementService in the test cases . Assert that relevant ports become available .
2007 - 05 - 07 Jonathan Pobst < monkey @ jpobst . com >
2007 - 04 - 10 Sebastien Pouliot < sebastien @ ximian . com >
[ build ] Make exclusion filename consistent between library and test
New test .
Update Wait and WaitAny with event disposal and single execution flow
[ util ] When creating the TLE exception only provide aditional data if both type and assembly names are provided . Otherwise one would fail and we ' d get an OOM instead .
2008 - 06 - 27 Ivan N . Zlatev < contact @ i - nz . net >
2005 - 05 - 11 Atsushi Enomoto < atsushi @ ximian . com >
Error statement support
Disable direct x86 aot tailcalls . Except to self . ( )
implementation changed to gdi +
Emit an explicit null check for some calls made out from gshared code .
[ profiler ] Remove some indirection in setting up GC events .
[ corlib ] Add missing public type
Minor changes in the script .
[ sgen ] Tiny code cleanup .
2007 - 01 - 23 Dick Porter < dick @ ximian . com >
new test
Increase timeout for some task tests
Explain the why ' s of THREAD_INFO_TYPE .
2005 - 05 - 10 Gonzalo Paniagua Javier < gonzalo @ ximian . com >
2003 - 09 - 25 Gonzalo Paniagua Javier < gonzalo @ ximian . com >
Fix minizip compile on Haiku ( )
Fix stream leak when failing to open a zip file
[ mcs ] Put back future lang version for easier migration path
2003 - 05 - 20 Atsushi Enomoto < ginga @ kit . hi - ho . ne . jp >
2005 - 02 - 18 Lluis Sanchez Gual < lluis @ novell . com >
2006 - 01 - 16 Sebastien Pouliot < sebastien @ ximian . com >
Updated .
2004 - 04 - 08 Zoltan Varga < vargaz @ freemail . hu >
Fix a warning .
[ arm64 ] Bump instruction sizes for atomic ops to match the emmited code
* ARM - related changes .
2003 - 06 - 10 Nick Drochak < ndrochak @ gol . com >
- Fixed accessibility level for Idle handler
2005 - 02 - 17 Atsushi Enomoto < atsushi @ ximian . com >
* ExtensionManager . cs : Support more than one XmlFormatExtensionPrefixAttribute
A small addition to Demo2 . cs
New test .
New test .
[ aot ] Handle gsharedvt versions of delegate invoke wrappers correctly , there could be multiple copies of them for the same class . Fixes .
ChangeLog : Updated ChaneLog .
* Monodoc / provider . cs : Add a RootTree . AddSource ( string ) method , so that
2007 - 08 - 30 Jonathan Pobst < monkey @ jpobst . com >
[ mcs ] Better error reporting after delegate probing failed . Fixs
[ asp . net ] Fix for bug . BuildManager must not fail with BuildProviders lacking the BuildProviderAppliesTo attribute .
Add guard against recursive base class with generic interfaces . Fixes
2005 - 06 - 12 Zoltan Varga < vargaz @ freemail . hu >
Fix 4 . 5 toolpath .
[ SafeHandle ] Protect Close against asynchronous exceptions
[ System ] Fixed RemoveConnectionGroup concurrent issue .
2003 - 10 - 05 Zoltan Varga < vargaz @ freemail . hu >
Forgot to turn off some debug output
2009 - 09 - 19 Sebastien Pouliot < sebastien @ ximian . com >
* OutputCacheModule . cs : Dont cache items when trace is enabled .
Use Path . Combine instead of concat
patch by Michael Hutchinson to use the environment variable http_proxy
2005 - 03 - 08 Zoltan Varga < vargaz @ freemail . hu >
2003 - 10 - 25 Atsushi Enomoto < ginga @ kit . hi - ho . ne . jp >
2004 - 02 - 27 Miguel de Icaza < miguel @ ximian . com >
Merge pull request from joncham / bug - null - interface - bitset
2006 - 09 - 02 Alexander Olk < alex . olk @ googlemail . com >
Process . PagedMemorySize64 now returns value from / proc / PID / status
* * * * Merged r65136 from HEAD * * * *
2002 - 02 - 12 Nick Drochak < ndrochak @ gol . com >
Fix the profile conformancy version
* MethodDef . cs ( MethodDef . AddLabel ) : Fix exception message .
2006 - 01 - 16 Zoltan Varga < vargaz @ gmail . com >
* scripts / script . in : Add @ framework_version @ blank .
2002 - 06 - 05 Gonzalo Paniagua Javier < gonzalo @ ximian . com >
[ System ] Process . Exited is now thread safe .
* XmlSerializerTests . cs : Undo some of Hagit ' s code formatting changes .
2006 - 05 - 06 Zoltan Varga < vargaz @ gmail . com >
2007 - 04 - 03 Jonathan Pobst < monkey @ jpobst . com >
New tests .
Fixed Windows MinGW build and avoid leakage of io - layer types .
* * * * Merged r43923 from MCS * * * *
2007 - 10 - 19 Nagappan A < anagappan @ novell . com >
2007 - 07 - 13 Marek Safar < marek . safar @ gmail . com >
[ Sys . Xml ] Add unit test for Novell bug
Tue Jan 23 17 : 48 : 42 CET 2007 Paolo Molaro < lupus @ ximian . com >
Bump roslyn to rc3
last patch broke the build reverting
Remove ancient todo item ( there might be others )
2006 - 03 - 20 Senganal T < tsenganal @ novell . com >
2002 - 12 - 16 Sebastien Pouliot < spouliot @ videotron . ca >
Mon Jul 12 17 : 47 : 00 CEST 2004 Paolo Molaro < lupus @ ximian . com >
2009 - 01 - 02 Ivan N . Zlatev < contact @ i - nz . net >
2010 - 06 - 02 Marek Safar < marek . safar @ gmail . com >
* SortedListTest . cs : Fixed deserialization test to pass on MS .
[ interp ] Add some missing icall signatures
Add gitter badge .
2002 - 10 - 09 Daniel Morgan < danmorg @ sc . rr . com >
2008 - 11 - 17 Zoltan Varga < vargaz @ gmail . com >
update to changes in compiler and class libraries
Clean up . o files from System . Native during ` make dist `
* Makefile : remove README , Mono . System . XML . csproj ,
Some frames were skipped in MonoProxy but we don ' t skip the frame_id , doing this the call_stack has a different position than it ' s in the real stack on mono side . ( )
* Added missing DashStyle property to Pen
Fixing some issues with the decimal parser / formatter ( dotnet / coreclr )
updated
SortedList entry .
2001 - 08 - 29 Ravi Pratap < ravi @ che . iitm . ac . in >
Add System . ComponentModel . DisplayAttribute
Improve with bug 49487 info
Make default be Xr
2003 - 07 - 17 Peter Williams < peter @ newton . cx >
Remove some obsolete methods
2002 - 09 - 24 Nick Drochak < ndrochak @ gol . com >
* * * empty log message * * *
* TraceData . cs : Fix control position when adding controls recursively .
[ asp . net ] Part of fix for bug . AsyncPostBackTrigger must react to events generated by the trigger control ' s children .
Fix windows build issue due to missing export in def files .
2010 - 07 - 14 Sebastien Pouliot < sebastien @ ximian . com >
Some findings
2006 - 04 - 07 Atsushi Enomoto < atsushi @ ximian . com >
Fix
Update
Correct definitions of CONV_R_UN and a couple of LCONVs
2009 - 10 - 12 Marek Habersack < mhabersack @ novell . com >
2008 - 06 - 08 Miguel de Icaza < miguel @ novell . com >
2003 - 07 - 30 Ben Maurer < bmaurer @ users . sourceforge . net >
Improve test
2003 - 09 - 04 Zoltan Varga < vargaz @ freemail . hu >
Merge pull request from ludovic - henry / monoerror - mono_string_get_pinned
2005 - 01 - 25 Sebastien Pouliot < sebastien @ ximian . com >
Initialize new handle chunks on in mono_handle_stack_alloc ( )
[ hybrid ] Enable COM , Remoting and AppDomains runtime support
[ sgen ] Increase parallelization of minors
Thu Sep 27 19 : 52 : 11 CEST 2001 Paolo Molaro < lupus @ ximian . com >
Resyncing ByteFX . Data with latest from ByteFX
Revert last 2 chances
Flush ( work in progress )
[ corlib ] Construct specific culture using most specific culture name region . Fixes
2007 - 05 - 30 Atsushi Enomoto < atsushi @ ximian . com >
- Updated code to match new HandleData . DeviceContext type
[ runtime ] Implement and enable the new interrupt code on amd64 .
2004 - 03 - 21 Gonzalo Paniagua Javier < gonzalo @ ximian . com >
Improve io - layer error logging .
enable the bindingsource tests
[ runtime ] Fix make install when boehm is disabled .
* PEAPI . cs : Add GenericTypeSpec so generic members can be
2009 - 03 - 26 Sebastien Pouliot < sebastien @ ximian . com >
2005 - 11 - 24 Miguel de Icaza < miguel @ novell . com >
2003 - 07 - 22 Zoltan Varga < vargaz @ freemail . hu >
2005 - 12 - 13 Atsushi Enomoto < atsushi @ ximian . com >
ifdef TARGET_JVM
Fix the ARM build .
Alter # if BIGENDIAN code with run - time condition .
[ acceptance - tests ] Bump coreclr to pick up 65b122151adb6b5ef7c27d8e0c268ef479fbe8b4
[ System ] Don ' t send response on empty stream write . Fixes
2008 - 04 - 22 Zoltan Varga < vargaz @ gmail . com >
[ scripts ] Don ' t use saxutils in mono - heapviz .
Correct code style problems .
2006 - 03 - 28 Atsushi Enomoto < atsushi @ ximian . com >
2004 - 06 - 16 Ben Maurer < bmaurer @ ximian . com >
Add tab classes to the build
[ wasm / aot ] Sprinkle some mono_runtime_get_no_exec to make sure we don ' t alloc objects during AOT .
2008 - 07 - 07 Ivan N . Zlatev < contact @ i - nz . net >
* get . c / h : New function to stringify function pointers .
2004 - 03 - 26 Sebastien Pouliot < sebastien @ ximian . com >
Avoid passing class pointers to some marshal icalls since its not aot compatible . Fixes .
[ profiler ] Acquire the exclusive buffer lock after the suspend lock .
Restore typecheck to corlib version check , so that mixing
[ profiler ] Don ' t link against libmono .
Mon Oct 6 16 : 10 : 02 CEST 2008 Paolo Molaro < lupus @ ximian . com >
2007 - 11 - 20 Atsushi Enomoto < atsushi @ ximian . com >
2005 - 05 - 04 Martin Baulig < martin @ ximian . com >
Tue Aug 3 11 : 19 : 07 CEST 2004 Paolo Molaro < lupus @ ximian . com >
Add missing changelog entry .
Set MONO_PATH while running the profiler tests .
Add missing null check to Directory enumerators
[ wasm ] use options directly in other places
Added a statusbar demo
2005 - 01 - 18 Zoltan Varga < vargaz @ freemail . hu >
[ asp . net ] The test must always run with the en - US locale
* XmlCDataSection . cs : Initial implementation .
Pass - relocation - model = static to llc in static mode .
In ilasm / tests :
Loading of nested type of missing type needs to create another missing type . Fixes
Merge pull request from akoeplinger / fix - xmlreader
Merge pull request from radical / async - local - fixes
Merge pull request from kumpera / handle_eloop_errno
2009 - 04 - 21 Sebastien Pouliot < sebastien @ ximian . com >
we pass those
fix typo
forgot this one
Small patch of love to MIS
2010 - 03 - 06 Rodrigo Kumpera < rkumpera @ novell . com >
2007 - 11 - 10 Marek Habersack < mhabersack @ novell . com >
2005 - 05 - 22 Zoltan Varga < vargaz @ freemail . hu >
Add test for bug 694908 .
2006 - 04 - 25 Zoltan Varga < vargaz @ gmail . com >
2004 - 11 - 26 Atsushi Enomoto < atsushi @ ximian . com >
Add documentation for exception traces .
2009 - 04 - 14 Zoltan Varga < vargaz @ gmail . com >
Fix the build
* System . EnterpriseServices . dll . sources : Added IConfigurationAttribute
Turn off ICU by default as planned
2007 - 11 - 12 Marek Habersack < mhabersack @ novell . com >
Small bug fix , to render better an acurrate error
Fix OP_LCONV_TO_R_UN_2 on x86 + aot . Fixes .
add test for a < clear / > element with an illegal property
Fix non - 4 . 0 build .
fix for bug 29365
EventSource : Add WriteEvent overloads , properties , constructors , to better match what ' s available in . NET 4 . 5
NUMA , SMP , RCU support ( just to get SCO go crazy ) . Typo fix .
* interpreter . cs ( Interpreter . GenerateMatch ) : Avoid allocating two
Added tests for Ajax client side functionality using Selenium infrastructure .
Fix host object existence check .
Fix the win64 build .
Use base type parameters to extract params modifier of virtual method
2006 - 10 - 25 Andreia Gaita < shana . ufie @ gmail . com >
Fix last comment
fix for the previous fix
2009 - 11 - 03 Rodrigo Kumpera < rkumpera @ novell . com >
* TypeDescriptorTests : Removed Ivan ' s local Category .
[ runtime ] Open addressing for MonoGHashTable
2010 - 01 - 07 Zoltan Varga < vargaz @ gmail . com >
[ sgen ] Remove ` SGEN_HAVE_CONCURRENT_MARK ` .
Add this for backwards compatibility
[ sgen ] Add / fix some assertions .
* expression . cs ( Binary . Operator ) : Renamed LogicalAnd / LogicalOr to
added new test for my latest ModuleBuilde . cs change
[ linker ] Fix monotouch tests which need Xamarin . iOS . dll ( )
2005 - 04 - 16 Ben Maurer < bmaurer @ ximian . com >
Add missing files from last commit
Add an ' mtriple ' aot option to set the target triple used by the AOT compiler in LLVM mode .
2009 - 06 - 21 Zoltan Varga < vargaz @ gmail . com >
2004 - 04 - 30 Sebastien Pouliot < sebastien @ ximian . com >
Contributed under the terms of the MIT / X11 license by
2008 - 10 - 02 Mark Probst < mark . probst @ gmail . com >
2008 - 11 - 10 Gonzalo Paniagua Javier < gonzalo @ novell . com >
[ wasm ] [ debugger ] Fix ` evaluateOnCallFrame ` for valuetype locals ( )
2005 - 03 - 11 Gonzalo Paniagua Javier < gonzalo @ ximian . com >
Prepping for monodoc move from monodoc / engine to mcs / class / monodoc . . .
* System . Windows . Forms / HtmlElement . cs : Small code cleanup
2009 - 06 - 29 Atsushi Enomoto < atsushi @ ximian . com >
* KeyInfoX509DataTest . cs
2007 - 05 - 01 Zoltan Varga < vargaz @ gmail . com >
2006 - 12 - 04 Chris Toshok < toshok @ ximian . com >
Merge pull request from madewokherd / byrefvariants
2008 - 10 - 06 Zoltan Varga < vargaz @ gmail . com >
* ComplexDataStructure . cs : Added MS product feedback bug ID to test
[ Microsoft . Build . Engine ] Simplify bd70a55ec7aaeb40b2b0054e0f162c709085842f
2009 - 09 - 17 Atsushi Enomoto < atsushi @ ximian . com >
Fix handling of fp arguments in the LLVM backend , they were copied using OP_MOVE , leading to various problems .
[ x86 ] Increase some instruction lengths .
[ monodoc ] Add a simple stack - based generic argument count method .
2007 - 01 - 17 Sebastien Pouliot < sebastien @ ximian . com >
* Binding . cs : Support < Property > IsNull checks . Also bind to the
2008 - 03 - 13 Marek Safar < marek . safar @ gmail . com >
Patch for approved by lupus
Thu Feb 8 12 : 46 : 18 CET 2007 Paolo Molaro < lupus @ ximian . com >
[ coop ] Handle blocking in parked threads correctly .
* TemplateControl . jvm . cs : Don ' t assume that controls containing page - specific
2005 - 11 - 25 Sebastien Pouliot < sebastien @ ximian . com >
[ runtime ] Use MonoError for new_thread_with_internal
Seal non - visible classes .
Revert " Apple has introduced the System Integrety Protection in OS X 10 . 11 . This prevents writes to / usr amongst other paths . / usr / local is still writable though . "
remove a couple of MonoTODO ' s
Allow external memory management of nodes .
Implement Task : : LongRunning option
2010 - 03 - 24 Jérémie Laval < jeremie . laval @ gmail . com >
[ sgen ] Bugfix : When scanning mod table , follow major - > major references .
[ interp ] Add x86 support ( )
minor changes in the Makefile
2004 - 05 - 05 Atsushi Enomoto < atsushi @ ximian . com >
Do not build docs if cross compiling or the mcs build is disabled
2008 - 05 - 06 Marek Safar < marek . safar @ gmail . com >
2004 - 12 - 04 Atsushi Enomoto < atsushi @ ximian . com >
2008 - 04 - 30 Jb Evain < jbevain @ novell . com >
[ System ] Don ' t throw SocketExceptions when we ' re stopping .
Implement memory barriers for amd64 / x86 . Fixes .
* Style . cs : performance optimization : suppress finalizer added
Flush
2010 - 06 - 22 Marek Safar < marek . safar @ gmail . com >
2009 - 11 - 30 Marek Safar < marek . safar @ gmail . com >
2007 - 09 - 17 Jonathan Pobst < monkey @ jpobst . com >
[ mono ] [ interp ] Optimize box + unbox to no - op ( )
these commits will add the full Microsoft . JScript public API . Now only last to fill in the blanks : - )
Remove these tests , they are no longer valid
* MapCodeGenerator . cs , SoapCodeExporter . cs : Generate the same XmlInclude
[ mkbundle ] Remove Mono . Posix , ICSharpCode . SharpZipLib deps
[ runtime ] Use AC_FUNC_STRERROR_R ( ) in Mono . Posix
2010 - 01 - 04 Zoltan Varga < vargaz @ gmail . com >
2009 - 02 - 03 Miguel de Icaza < miguel @ novell . com >
[ cleanup ] WebConfigurationManager : remove hasConfigErrors
Remove a few braces that are contrary to our prescribed style . ( )
* TreeView . cs : Calculate plus minus bounds when there is a
Fix tests where a DSA CA certificate expired at the end of 2011
Sat Feb 16 12 : 41 : 41 CET 2002 Paolo Molaro < lupus @ ximian . com >
[ amd64 ] Exclude the GSHAREDVT tests as well since they don ' t work yet .
Patch .
[ sgen ] Finish the concurrent collection only on the gc thread
2009 - 01 - 20 Atsushi Enomoto < atsushi @ ximian . com >
Install utils / { mono - dl . h , mono - compiler . h }
* MenuAPI . cs :
2008 - 08 - 18 Jonathan Pobst < monkey @ jpobst . com >
cvs ignores , sorry about the spam
* DirectoryTest . cs : Fixed compiler warning by improving test .
Share more common code
Fix incorrect Debug . Assert statements
Update sdks / wasm / framework / src / WebAssembly . Bindings / Core / Array . cs
Merge pull request from pruiz / xamarin - bug - 4108
* * * empty log message * * *
* main . c ( dis_genericparam ) : oops , fix off - by - one error .
[ runtime ] Free cached_info tables on cleanup
2003 - 05 - 12 Dick Porter < dick @ ximian . com >
2008 - 11 - 25 Gonzalo Paniagua Javier < gonzalo @ novell . com >
2009 - 01 - 01 Ivan N . Zlatev < contact @ i - nz . net >
* Control . cs : ControlCollection . Add : Remove a couple of duplicated
Fix release semantics to make sure it flushes the store buffer on ia32 / amd64 .
2006 - 01 - 19 Peter Dennis Bartok < pbartok @ novell . com >
One - liner in the comments of decimal2string ( )
2007 - 05 - 11 Jeffrey Stedfast < fejj @ novell . com >
2002 - 06 - 09 Martin Baulig < martin @ gnome . org >
- Added Idle event
Add new ( 2 . 0 ) System . Uri related classes and a lot of tests ( unit and CAS ) for them
Warn on failure to create default Reflection . Emit SymbolWriter instead of throwing and failing ( )
[ ci ] Compile the class lib tests in a separate step
ImageButton . cs : implemented GenerateEmptyAlternateText
Fix that corrects a mishandling of UInt64 ' s .
pie sample
* * * * Merged from MCS * * * *
New tests .
Add testcase for pre - escaped strings preceded by non - english characters .
2005 - 05 - 20 Peter Bartok < pbartok @ novell . com >
[ sgen ] Direct card table implementation .
Add new sample error case , from Dan
Fix .
* Makefile ( NO_TEST ) : Set .
Fix a loader leak in mono_class_create_from_typedef .
2008 - 08 - 12 Zoltan Varga < vargaz @ gmail . com >
Mono friendly string
* TestIcon . cs : Add test case for bug .
2008 - 06 - 14 Zoltan Varga < vargaz @ gmail . com >
First batch of changes
2005 - 06 - 15 Ben Maurer < bmaurer @ ximian . com >
Upgraded to beta 2
[ runtime ] Avoid recreating the icall table in lookup_icall_symbol ( ) . ( )
* Stack . cs : Clone ( ) doesn ' t need to check if it ' s synchronized , since
2007 - 07 - 23 Jonathan Pobst < monkey @ jpobst . com >
2004 - 12 - 11 Gonzalo Paniagua Javier < gonzalo @ ximian . com >
2005 - 01 - 21 Lluis Sanchez Gual < lluis @ novell . com >
Merge pull request from BrzVlad / fix - binprot - stats
Ship Mono . Cecil as well
typo on last commit
2007 - 12 - 19 Jonathan Pobst < monkey @ jpobst . com >
New Attribute to compile list added
2002 - 04 - 15 Patrik Torstensson < patrik . torstensson @ labs2 . com >
Fix to JAVAEE TypeAs Expression .
2010 - 04 - 07 Gonzalo Paniagua Javier < gonzalo @ novell . com >
2009 - 01 - 14 Zoltan Varga < vargaz @ gmail . com >
Print a normal stack trace even if a hard stack overflow happens in unmanaged code .
Wed Jun 8 19 : 26 : 38 CEST 2005 Paolo Molaro < lupus @ ximian . com >
2005 - 10 - 25 Gonzalo Paniagua Javier < gonzalo @ ximian . com >
2009 - 01 - 04 Rodrigo Kumpera < rkumpera @ novell . com >
TimeZoneInfo . Local must be cached , otherwise reference comparisons fail later on .
Allow forcing HTTP protocol version
2005 - 05 - 11 Geoff Norton < gnorton @ customerdna . com >
test themes
Remove monoburg , it ' s not used anymore .
( monoone_DATA ) : Add mcs . exe . config .
Simplify Parallel . For execution .
2009 - 09 - 22 Sebastien Pouliot < sebastien @ ximian . com >
2006 - 12 - 04 Marek Sieradzki < marek . sieradzki @ gmail . com >
2002 - 08 - 09 Tim Coleman < tim @ timcoleman . com >
[ mini ] Expand clr - memory - model effect ( )
[ Facades ] Use Open . snk for netstandard . dll
Add class ActiveDirectorySiteLinkBridge , ActiveDirectorySiteLinkCollection
- Removed VirtualKeys to XplatUIStructs
2010 - 03 - 17 Atsushi Enomoto < atsushi @ ximian . com >
2006 - 03 - 09 Martin Baulig < martin @ ximian . com >
Add ` Signing . props ` file to ` common ` directory to be referenced from a common place
2007 - 08 - 27 Atsushi Enomoto < atsushi @ ximian . com >
2008 - 05 - 14 Jonathan Pobst < monkey @ jpobst . com >
2007 - 10 - 23 Geoff Norton < gnorton @ novell . com >
2008 - 12 - 08 Jb Evain < jbevain @ novell . com >
2005 - 05 - 15 Atsushi Enomoto < atsushi @ ximian . com >
[ Sys . Data . DSExtensions ] add AsEnumerable ( ) test .
Use proper sized integer for ioctlsocket on Win32 . ( )
* Control . cs ( PreProcessMessage ) : Moved menu handling back , but
2006 - 11 - 13 Gonzalo Paniagua Javier < gonzalo @ ximian . com >
2008 - 03 - 28 Marek Safar < marek . safar @ gmail . com >
2007 - 08 - 03 Zoltan Varga < vargaz @ gmail . com >
* assign . cs : Fix for implicit conversions in Compound Assignments
Add missing attribute classes
Add this too
2005 - 07 - 20 Atsushi Enomoto < atsushi @ ximian . com >
Set max fetch size for jdbc statement .
just in case sign extension kicks in
[ verifier ] Fail to merge stacks with different safe byref ' ness .
Add new unit tests for X501Name
mini - ppc . h : Undo typo of MONO_CONTEXT_SET_SP ( ppc doesn ' t define this
Use IDictionaryEnumerator in a loop where both key and value are used
Updated .
2008 - 04 - 30 Marek Safar < marek . safar @ gmail . com >
[ runtime ] Fixed leak of MonoListItems in MonoCQ
Fix Location to have reasonable limits when running under MD
Merge pull request from angeloc / patch - 1
New test .
2006 - 06 - 23 John Luke < john . luke @ gmail . com >
Mon Jan 19 17 : 52 : 33 CET 2004 Paolo Molaro < lupus @ ximian . com >
2006 - 02 - 07 Chris Toshok < toshok @ ximian . com >
CodeManager cleanup : ( )
Rename MONO_GC_REGISTER_ROOT ( ) - > MONO_GC_REGISTER_ROOT_PINNING ( ) .
2004 - 10 - 14 Atsushi Enomoto < atsushi @ ximian . com >
2004 - 07 - 28 Dick Porter < dick @ ximian . com >
2004 - 11 - 05 Miguel de Icaza < miguel @ ximian . com >
Fix another AOT crash if a signature cannot be loaded .
2006 - 03 - 14 Zoltan Varga < vargaz @ gmail . com >
* XplatUIWin32 . cs : Changed Win32CreateWindow to take enums instead of
More link fixage
Rename a few fields .
2007 - 05 - 01 Everaldo Canuto < everaldo @ simios . org >
Wed Aug 8 15 : 30 : 05 CEST 2001 Paolo Molaro < lupus @ ximian . com >
2009 - 05 - 22 Atsushi Enomoto < atsushi @ ximian . com >
* Mono . Documentation / monodocer . cs : Cleanup : turn static data into
2007 - 01 - 19 Sebastien Pouliot < sebastien @ ximian . com >
2004 - 09 - 24 Umadevi S < sumadevi @ novell . com >
Add to the build
New tests for Read and Write APIs .
[ acceptance - tests ] Disable the msbiology test completely .
2007 - 11 - 07 Juraj Skripsky < js @ hotfeet . ch >
2009 - 08 - 18 Christian Hergert < chris @ dronelabs . com >
2002 - 11 - 04 Tim Coleman ( tim @ timcoleman . com )
[ sgen ] Move TLS emit macros to Mono - specific file .
2008 - 02 - 13 Zoltan Varga < vargaz @ gmail . com >
Fix some small memory leaks .
Fix InternalGetHashCode for boehm / null GC ( )
[ System ] Fix DnsTest . GetHostByName ( ) test .
2006 - 02 - 26 Marek Sieradzki < marek . sieradzki @ gmail . com >
2004 - 07 - 22 Dick Porter < dick @ ximian . com >
* AbsoluteToRelativePath : New .
2005 - 06 - 20 Martin Baulig < martin @ ximian . com >
2004 - 11 - 01 Gonzalo Paniagua Javier < gonzalo @ ximian . com >
Updated 2 . 0 . 3600 . 0 assembly versions to 2 . 0 . 0 . 0 .
New tests .
Make gshared variables GC tracked .
2002 - 05 - 13 Dan Lewis < dihlewis @ yahoo . co . uk >
[ interp ] Restore valuetype copying for pinvoke case .
2006 - 05 - 17 Zoltan Varga < vargaz @ gmail . com >
Do not import xs : any member as XmlElement [ ] . Create a wrapper class instead .
2009 - 05 - 15 Geoff Norton < gnorton @ novell . com >
2008 - 08 - 15 Carlos Alberto Cortez < calberto . cortez @ gmail . com >
Add missing files to the build
Incremented version number .
[ mini ] iltests . exe depends on TestDriver . dll ( )
2002 - 07 - 27 Andrew Birkett < andy @ nobugs . org >
* System . Design . dll . sources : added more stub for classes
Kill some dead code and fix a function name
Merge pull request from ludovic - henry / coop - threads - cleanup
Speed up suspending when running under the loader lock .
* OleDbConnection . cs : Derive from DbConnection on 2 . 0 profile , and
Documented MONO_SHARED_HOSTNAME
Fix a memory leak .
2008 - 11 - 03 Zoltan Varga < vargaz @ gmail . com >
2005 - 10 - 03 Atsushi Enomoto < atsushi @ ximian . com >
added
Add regression test for
Fix memory corruption in the cache cast wrappers .
2002 - 12 - 07 Ravi Pratap < ravi @ ximian . com >
2006 - 03 - 07 Carlos Alberto Cortez < calberto . cortez @ gmail . com >
[ runtime ] Managed ProcessModule now has versions .
Support IBM AIX on POWER ( )
2004 - 09 - 17 Zoltan Varga < vargaz @ freemail . hu >
2004 - 02 - 23 Sebastien Pouliot < sebastien @ ximian . com >
* Driver . cs : Report entry for each il file processed
[ jit ] Enable alias analysis dce even if there are no ldaddr instructions eliminated , they could have been eliminated earlier . Enable only for netcore for now .
Bump referencesource
* * * Merged revisions from mcs : 51705 , 51728
2005 - 12 - 05 Marek Safar < marek . safar @ seznam . cz >
2006 - 01 - 18 Martin Baulig < martin @ ximian . com >
[ sgen ] Implement a simple thread pool and do concurrent sweep with it .
2007 - 03 - 14 Chris Toshok < toshok @ ximian . com >
2004 - 02 - 10 Gonzalo Paniagua Javier < gonzalo @ ximian . com >
2003 - 07 - 07 Gonzalo Paniagua Javier < gonzalo @ ximian . com >
OSX won ' t expose IPV6_PKTINFO unless __APPLE_USE_RFC_3542 is defined .
Fix a problem with 9adc7023e66a04f162efa041ff2452222a85c977 . Implement constrained calls to enum interface methods correctly .
2005 - 09 - 07 Chris Toshok < toshok @ ximian . com >
2006 - 12 - 05 Chris Toshok < toshok @ ximian . com >
Revert no longer needed changes
2008 - 01 - 29 Zoltan Varga < vargaz @ gmail . com >
2006 - 10 - 25 Chris Toshok < toshok @ ximian . com >
2003 - 08 - 28 Martin Baulig < martin @ ximian . com >
Disable wrapper generation code when DISABLE_JIT is defined .
[ tests ] Different CPUs have different precision
2006 - 04 - 26 Miguel de Icaza < miguel @ novell . com >
formatting
[ corlib ] Disable a ParameterInfoTest test on MONOTOUCH
Gave up finding " correct " way to deserialize XamlType and made it special case : (
* HttpSimpleClientProtocol . cs , HttpWebClientProtocol . cs ,
Changed Uri . Unescape to use UriHelper . FormatAbsolute .
Update profiles to use - d : and make generics build use NET_1_1
2005 - 05 - 25 Lluis Sanchez Gual < lluis @ novell . com >
Make asserts more usefull .
2008 - 01 - 31 Jb Evain < jbevain @ novell . com >
* FieldBuilder . cs : fixed implementation to match MS . NET ,
* Mono . Documentation / ManifestResourceResolver . cs ,
* grp . c , pwd . c : Clear errno before calling the actual calls , as errno may
Try this as well .
Convert null attribute type to Attribute . Fixes
[ remoting ] call TransparentProxy : : StoreRemoteField directly from STFLD_REMOTE wrapper .
[ rpm spec ] Mono . Web and OpenSystem . C do not exist anymore
2007 - 04 - 20 Atsushi Enomoto < atsushi @ ximian . com >
Remove a warning and allow eglib to build with warnings .
Beta OracleClient implementation
2010 - 01 - 03 Zoltan Varga < vargaz @ gmail . com >
Tue Sep 25 19 : 03 : 52 CEST 2001 Paolo Molaro < lupus @ ximian . com >
convert ArrayList to array only in case of DeserializeObject ( )
2008 - 05 - 19 Zoltan Varga < vargaz @ gmail . com >
threadpool . c ( async_invoke_thread ) :
2010 - 04 - 29 Zoltan Varga < vargaz @ gmail . com >
2008 - 12 - 05 Ivan N . Zlatev < contact @ i - nz . net >
2007 - 09 - 12 Marek Safar < marek . safar @ gmail . com >
In System . Threading :
2008 - 08 - 26 Miguel de Icaza < miguel @ novell . com >
2009 - 10 - 02 Marek Safar < marek . safar @ gmail . com >
* Application . cs : In ProductVersion first try AssemblyFileVersion
2004 - 03 - 16 Andreas Nahr < ClassDevelopment @ A - SoftTech . com >
2008 - 07 - 27 Zoltan Varga < vargaz @ gmail . com >
* Image . cs : Write Sections
2002 - 10 - 26 DennisHayes < dennish @ raytek . com >
* Hwnd . cs : Don ' t compute the menu size for
2002 - 12 - 20 Gonzalo Paniagua Javier < gonzalo @ ximian . com >
[ runtime ] Switch the order of LLVM_LIBS and LLVM_LDFLAGS on the linker command line . Hopefully fixes .
2003 - 02 - 17 Gonzalo Paniagua Javier < gonzalo @ ximian . com >
Reject inaccessible members based on declaring type in dynamic context
fix crlf
WebTest . cs redesigned to work with NunitWeb .
2006 - 07 - 19 Zoltan Varga < vargaz @ gmail . com >
Method field count ( )
[ profiler ] Use macro constants instead of passing a bunch of bools .
2005 - 08 - 25 Atsushi Enomoto < atsushi @ ximian . com >
[ msbuild ] Add option to nuget - hash - extractor to generate a list of the
2004 - 06 - 24 Atsushi Enomoto < atsushi @ ximian . com >
2006 - 12 - 17 Yoni Klain < yonik @ mainsoft . com >
Fixes build .
replace
2005 - 11 - 11 Dick Porter < dick @ ximian . com >
2002 - 12 - 21 Miguel de Icaza < miguel @ ximian . com >
[ packaging / msbuild ] Bump to get new changes on xplat - master
remove useless stuff
More API cleanups and documentation .
2004 - 03 - 23 Martin Baulig < martin @ ximian . com >
Merge pull request from lambdageek / dev / marshal - fix
Implement correct DataSourceID lookup
2007 - 11 - 17 Miguel de Icaza < miguel @ novell . com >
[ jit ] Replace usage of mini_type_get_underlying_type with first arg null with new function mini_replace_type .
[ Mono . Debugger . Soft ] Move TypeLoadClass types to new file to improve testing
2008 - 12 - 03 Rodrigo Kumpera < rkumpera @ novell . com >
2003 - 04 - 14 Gaurav Vaish < gvaish_mono AT lycos . com >
2002 / 03 / 08 Nick Drochak < ndrochak @ gol . com >
2008 - 03 - 07 Zoltan Varga < vargaz @ gmail . com >
2008 - 08 - 08 Zoltan Varga < vargaz @ gmail . com >
2005 - 10 - 24 Martin Baulig < martin @ ximian . com >
* TextBoxBase . cs : We need to cap this value , since Maximum -
[ merp ] Capture Environment . FailFast message in crash report ( )
[ interp ] add another signature for interp_in wrappers
2004 - 01 - 20 Gonzalo Paniagua Javier < gonzalo @ ximian . com >
Check type parameter substitutions for nested interfaces as well
2006 - 01 - 13 Chris Toshok < toshok @ ximian . com >
2006 - 11 - 27 Igor Zelmanovich < igorz @ mainsoft . com >
2002 - 08 - 05 Martin Baulig < martin @ gnome . org >
Sources . . .
* xbuild / Microsoft . Common . targets : Emit list of files written
Apply a workaround for a gcc 4 . 6 problem on arm .
Fix parse errors in GetRelativePath and add to the ErrorCount when one occurs so the build will fail
2002 - 08 - 01 Martin Baulig < martin @ gnome . org >
[ sgen ] Separate descriptor type enums .
2007 - 01 - 25 Atsushi Enomoto < atsushi @ ximian . com >
[ runtime ] Fix MonoError lifecycle in do_try_exec_main ( )
* build - csproj2k5 , SWF2k5 . csproj : add System . configuration reference
This is a clean up of last commit . One line had spaces where there should
2006 - 09 - 02 Zoltan Varga < vargaz @ gmail . com >
2003 - 01 - 20 Gonzalo Paniagua Javier < gonzalo @ ximian . com >
2007 - 01 - 25 Chris Toshok < toshok @ ximian . com >
2005 - 11 - 12 Pedro Martínez Juliá < pedromj @ gmail . com >
[ mcs ] Handle partial entry point method
Fix a warning .
2003 - 08 - 04 Martin Baulig < martin @ ximian . com >
Wrong date in the changelog
2002 - 08 - 06 Tim Coleman < tim @ timcoleman . com >
tons more corcompare work
2004 - 01 - 30 Sebastien Pouliot < spouliot @ videotron . ca >
[ build ] Remove PLATFORM_CORLIB from platform Makefiles
Added samples directory with notepad . cs as a visual test and sample for scrolblar , multiline text , file save dialog , and menus .
RectangleF . cs : Fix edge intersection . Path by Brian Browning . [ Fix bug ]
2004 - 07 - 20 Anirban Bhattacharjee < banirban @ novell . com >
[ mini ] Allow stind . ref only on object references . ( )
2006 - 10 - 24 Alexander Olk < alex . olk @ googlemail . com >
[ bcl ] Add more reflection types to monotouch
2008 - 10 - 21 Jonathan Pobst < monkey @ jpobst . com >
Grand finalle to this long patch series . Move remembered set implementation to use a vtable - like schema .
* RemotingProxy . cs : Changed the rules for casting WKO proxies , to match
* ILTokenizer . cs : Use the IsInstr method for checking if a string is an instruction
2004 - 03 - 19 Gonzalo Paniagua Javier < gonzalo @ ximian . com >
ChangeLog : Updated ChangeLog .
Seal non - visible classes .
In Test / System . Windows . Forms :
2008 - 01 - 25 Jb Evain < jbevain @ novell . com >
2002 - 01 - 18 Nick Drochak < ndrochak @ gol . com >
remove extra printf .
* ComboBox . cs : Handle Home and End keys as well as all combinations of
2006 - 05 - 18 Mike Kestner < mkestner @ novell . com >
2008 - 12 - 29 Sebastien Pouliot < sebastien @ ximian . com >
2008 - 07 - 25 Ivan N . Zlatev < contact @ i - nz . net >
2007 - 11 - 02 Atsushi Enomoto < atsushi @ ximian . com >
Indentation
Remove mentions of Logging and Heap Shot profilers from man page
Update license for samples
2007 - 12 - 28 Jonathan Pobst < monkey @ jpobst . com >
2009 - 12 - 17 Zoltan Varga < vargaz @ gmail . com >
2005 - 10 - 18 Jordi Mas i Hernandez < jordi @ ximian . com >
[ netcore ] Publish to BAR too ( )
2003 - 06 - 29 Gonzalo Paniagua Javier < gonzalo @ ximian . com >
2006 - 01 - 20 Chris Toshok < toshok @ ximian . com >
2008 - 08 - 11 Marek Habersack < mhabersack @ novell . com >
2003 - 12 - 08 Martin Baulig < martin @ ximian . com >
[ runtime ] Poll for interruption on cctor execution if mono_threads_end_abort_protected_block returns true .
2004 - 03 - 09 Atsushi Enomoto < atsushi @ ximian . com >
[ System . Core ] Linker friendly test update
2006 - 03 - 27 Dick Porter < dick @ ximian . com >
[ jit ] Use the mono_mutex types / functions of CriticalSection ' s .
2008 - 05 - 12 Atsushi Enomoto < atsushi @ ximian . com >
Keep only 1 Dispose implementation
[ runtime ] Avoid creating callbacks . interp_get_remoting_invoke if the interpreter is not enabled at runtime . ( )
[ netcore ] Restore Roslyn compilers from NuGet to allow up - to - date version to be used ( )
Build fix for the no gc case .
2004 - 06 - 11 Martin Baulig < martin @ ximian . com >
2007 - 03 - 08 Marek Sieradzki < marek . sieradzki @ gmail . com >
Revert " [ jit ] Remove the partial sharing enabling code , its always enabled now . "
2002 - 08 - 28 Dick Porter < dick @ ximian . com >
then eol - style = native .
2003 - 12 - 02 Gonzalo Paniagua Javier < gonzalo @ ximian . com >
[ sgen ] Have the worker state just be an integer .
Improve logging for EventSource . WriteEvent parameter mismatch ( )
* added [ Flags } attribute that I mistakenly removed before the first commit . *
* TextBoxBase . cs : When positioning the caret , use the line , pos
explicitly implement ICloneable . Clone as . Net does
[ mprof - report ] Add counter filter when not in verbose mode
2003 - 02 - 23 Peter Williams < peterw @ ximian . com >
Added some tests for the SelectNodes method .
Remove unnecessary file .
[ w32socket ] Translate some errno codes to WSA_ errors imprecisely .
* ThreadTest . cs : Added test for bug . Improved test for
2008 - 03 - 20 Carlos Alberto Cortez < calberto . cortez @ gmail . com >
[ runtime ] Add a new MonoGSharedGenericParam structure as an extension of the MonoGenericParamFull structure , put the ' parent ' field there .
2005 - 09 - 13 Chris Toshok < toshok @ ximian . com >
[ threads ] Implement hybrid cooperative suspend
Fix bcl - size - diff calculation ( )
forgot this
[ System ] [ tests ] Make sure Dns tests works in a mixed IPv4 / IPv6 network too
* main . c : Use escaped names for method , and type names . Initialize
2003 - 03 - 26 Zoltan Varga < vargaz @ freemail . hu >
2009 - 07 - 20 Geoff Norton < gnorton @ novell . com >
2002 - 10 - 14 Sebastien Pouliot < spouliot @ videotron . ca >
* * * empty log message * * *
Cleanup arm64 SeqPointInfo .
[ AOT ] remove confusing llcopts flag ( )
Add the compiler server log to the gitignore
Wed Feb 20 14 : 54 : 01 CET 2002 Paolo Molaro < lupus @ ximian . com >
* sn . cs : Attempt to load the strongnames . config file from the GAC
2006 - 09 - 18 Gonzalo Paniagua Javier < gonzalo @ ximian . com >
2009 - 02 - 07 Zoltan Varga < vargaz @ gmail . com >
* FileDialog . cs : Next try
Added mmap and madvise checks .
* MenuAPI . cs : On instance of MenuTracker check if source control is
2002 - 07 - 31 Martin Baulig < martin @ gnome . org >
Implement support for ProfileCommon class generation
2010 - 03 - 18 Marek Habersack < mhabersack @ novell . com >
2010 - 01 - 08 Rodrigo Kumpera < rkumpera @ novell . com >
2002 - 07 - 13 Gonzalo Paniagua Javier < gonzalo @ ximian . com >
Fix NRE for unknown method token for some generated methods
2007 - 04 - 06 Andreas Faerber < andreas . faerber @ web . de >
add com interop classes
Use simple Convert when storing registry values
2008 - 12 - 10 Atsushi Enomoto < atsushi @ ximian . com >
[ mcs ] Implements expression bodied members
2003 - 02 - 07 Martin Baulig < martin @ ximian . com >
Use mktemp instead of $ RANDOM to generate unique names
Merge pull request from esdrubal / mono - sym - enable
Mon Apr 22 19 : 44 : 16 CEST 2002 Paolo Molaro < lupus @ ximian . com >
Fixed how EnumMap serializes flag values . Fixes .
2006 - 08 - 13 Miguel de Icaza < miguel @ novell . com >
2004 - 06 - 08 Gonzalo Paniagua Javier < gonzalo @ ximian . com >
2003 - 11 - 18 Zoltan Varga < vargaz @ freemail . hu >
2008 - 02 - 26 Rodrigo Kumpera < rkumpera @ novell . com >
Override SetAllowDrop on X11 so an error message isn ' t printed
2008 - 06 - 17 Atsushi Enomoto < atsushi @ ximian . com >
* jit - icalls . c ( helper_compile_generic_method ) : Document the
2009 - 08 - 07 Atsushi Enomoto < atsushi @ ximian . com >
2010 - 03 - 06 Gonzalo Paniagua Javier < gonzalo @ novell . com >
2006 - 03 - 13 Peter Dennis Bartok < pbartok @ novell . com >
2007 - 07 - 05 Sebastien Pouliot < sebastien @ ximian . com >
2004 - 11 - 24 Atsushi Enomoto < atsushi @ ximian . com >
2001 - 12 - 23 Miguel de Icaza < miguel @ ximian . com >
Disable per - collection parallelism in debugger tests ( )
Fixitificate
2007 - 10 - 29 Atsushi Enomoto < atsushi @ ximian . com >
Fix a missing \ to resolve make clean in HEAD .
Bump reference source to fix the build
forgot to remove dead code
Remove CPP comments
[ interp ] Optimize ldobj / stobj ( )
2008 - 01 - 21 Jb Evain < jbevain @ novell . com >
+ * ServerType . cs : Do not use Activator . CreateInstance , use the call
Revert " Fix typo in commit b6d634dd . "
2008 - 01 - 16 Zoltan Varga < vargaz @ gmail . com >
Fix issue in NamedPipeServerStream . GetImpersonationUserName ( ) ( )
2008 - 11 - 04 Atsushi Enomoto < atsushi @ ximian . com >
2004 - 04 - 27 Zoltan Varga < vargaz @ freemail . hu >
Remove System . Net . Http from MT mobile build
2004 - 06 - 08 Marek Safar < marek . safar @ seznam . cz >
2010 - 04 - 01 Rodrigo Kumpera < rkumpera @ novell . com >
In mcs :
New test .
New test .
[ system . data ] Fixes mobile build
* Added enums delagtes interfaces and stubbed out classes . *
2002 - 05 - 30 Martin Baulig < martin @ gnome . org >
2009 - 07 - 24 Jb Evain < jbevain @ novell . com >
[ mono - symbolicate ] Created SeqPointInfo . cs
2002 - 06 - 08 Martin Baulig < martin @ gnome . org >
New tests .
Sat Mar 1 15 : 32 : 56 CET 2003 Paolo Molaro < lupus @ ximian . com >
Fix crasher
[ llvmonly ] Make sure that linkonce methods need no initialization since they could be called before the AOT module they belong to is loaded . Still not enabled , since they seem to actually increase executable size .
* MSBuild . cs ( Execute ) : Reset current directory back after every
2008 - 12 - 01 Ivan N . Zlatev < contact @ i - nz . net >
changelog
2004 - 05 - 18 Gonzalo Paniagua Javier < gonzalo @ ximian . com >
2004 - 04 - 16 Lluis Sanchez Gual < lluis @ ximian . com >
2004 - 03 - 08 Umadevi S ( sumadevi @ novell . com )
Add possibility of generating a runtime . nupkg for Mono ( Mac - only right now )
2008 - 07 - 25 Rodrigo Kumpera < rkumpera @ novell . com >
2004 - 08 - 29 Zoltan Varga < vargaz @ freemail . hu >
Update
* XplatUIX11 . cs : Create FosterParent with border width at 0 .
Compilation fixes .
2001 - 08 - 16 Alex Graveley < alex @ ximian . com >
[ corlib ] Exception stack traces now show method index if needed .
In class / Microsoft . Build . Engine / Microsoft . Build . BuildEngine :
Wed Jan 5 16 : 17 : 27 CET 2005 Paolo Molaro < lupus @ ximian . com >
2008 - 09 - 24 Zoltan Varga < vargaz @ gmail . com >
2005 - 09 - 23 Chris Toshok < toshok @ ximian . com >
2008 - 07 - 27 Zoltan Varga < vargaz @ gmail . com >
Look for resgen2 where mcs is .
Readme for mono hackers .
Document
Merge pull request from directhex / sgen_on_msvc
* Console . cs ( Readline ) [ NET_2_0 ] : Avoid " uninitialized variable " error .
2009 - 01 - 06 Jb Evain < jbevain @ novell . com >
2006 - 02 - 10 Martin Baulig < martin @ ximian . com >
2009 - 01 - 09 Zoltan Varga < vargaz @ gmail . com >
Mon Sep 11 13 : 10 : 12 CEST 2006 Paolo Molaro < lupus @ ximian . com >
2010 - 07 - 15 Marek Habersack < mhabersack @ novell . com >
2002 - 03 - 19 Miguel de Icaza < miguel @ ximian . com >
2010 - 03 - 31 Rodrigo Kumpera < rkumpera @ novell . com >
[ wasm ] Disable aot support in release builds . ( )
avoid creation of arrays
2010 - 04 - 16 Zoltan Varga < vargaz @ gmail . com >
Fixes for broken RO check
2008 - 12 - 05 Marek Habersack < mhabersack @ novell . com >
Wed May 8 16 : 40 : 50 CEST 2002 Paolo Molaro < lupus @ ximian . com >
2004 - 07 - 11 Atsushi Enomoto < atsushi @ ximian . com >
2008 - 02 - 29 Zoltan Varga < vargaz @ gmail . com >
Cleanups .
[ bcl ] Update BCL Linked Size
* ReflectionHelper . cs : In CheckSerializableType ( ) check that the type
Remove unused bit position constants in ReaderWriterLockSlim
Fix severe breakage of sgen bridge code . 2nd and 3rd step where not been called anymore .
Optimised the SwapInt and SwapLong methods
2007 - 06 - 12 Marek Safar < marek . safar @ gmail . com >
2010 - 06 - 07 Marek Safar < marek . safar @ gmail . com >
2006 - 08 - 28 Mike Kestner < mkestner @ novell . com >
[ HttpListener ] Avoid contention
[ runtime ] Add a signal handler for sigterm ( )
Avoid endianness swap decisions in the inner loop
[ profiler ] Start jotting down some format oddities we should eventually fix .
2005 - 04 - 08 Gonzalo Paniagua Javier < gonzalo @ ximian . com >
* TestPoint . cs : GetHashCode and ToString tests added
removed double topdir definition
2002 - 10 - 09 Gonzalo Paniagua Javier < gonzalo @ ximian . com >
* Splitter . cs : Make MaxSize internal so that UIA code can use it .
System . XML : Remove deprecated NET_2_0 preprocessor conditionals
This was only for debugging .
2004 - 02 - 21 Atsushi Enomoto < atsushi @ ximian . com >
DirectoryEntry . cs :
2004 - 03 - 20 Sebastien Pouliot < sebastien @ ximian . com >
Fix another [ 0 ] issue .
[ mscorlib ] Marshal . GetExceptionForHR ( ) takes an ` errorInfo ` parameter .
2004 - 07 - 24 Zoltan Varga < vargaz @ freemail . hu >
2003 - 09 - 21 eran < erand @ mainsoft . com >
2008 - 03 - 10 Rodrigo Kumpera < rkumpera @ novell . com >
Enhanced test for valuetypes .
2006 - 06 - 06 Zoltan Varga < vargaz @ gmail . com >
2004 - 09 - 03 Tim Coleman ( tim @ timcoleman . com )
Fix the brokeness that I introduced earlier
[ arm ] Fix support for soft breakpoints .
2004 - 05 - 07 Atsushi Enomoto < atsushi @ ximian . com >
[ mini ] return gracefully from compile_special if a method can not be compiled
Implements checking of referenced members with missing references
2003 - 07 - 28 Piers Haken < piersh @ friskit . com >
* macros . c : Use gint64 , not long , to match C # declaration . Fixes 71152 .
[ arm ] Fix the DISABLE_JIT build .
2002 - 10 - 25 Daniel Morgan < danmorg @ sc . rr . com >
Merge pull request from saper / unit - tests - static
more tests
2008 - 08 - 24 Ivan N . Zlatev < contact @ i - nz . net >
2003 - 07 - 13 Zoltan Varga < vargaz @ freemail . hu >
[ update - ecma - xml ] Generate the / / Type / @ FullNameSP attribute if not present .
Last commit was not entirely correct . . .
Fix for compilation in TARGET_JVM
Code style police
2005 - 03 - 18 Stefan Buehler < sbuehler @ gmx . ch >
* ComboBoxTests . cs , ComboBoxTest . cs : Unified ComboBoxTest . cs and
2010 - 03 - 24 Atsushi Enomoto < atsushi @ ximian . com >
2009 - 09 - 10 Rodrigo Kumpera < rkumpera @ novell . com >
2004 - 01 - 12 Atsushi Enomoto < atsushi @ ximian . com >
2008 - 06 - 10 Rodrigo Kumpera < rkumpera @ novell . com >
Re - enable tests after change to compiler - tester
2009 - 07 - 10 Gonzalo Paniagua Javier < gonzalo @ novell . com >
2004 - 08 - 15 Zoltan Varga < vargaz @ freemail . hu >
[ jit ] Clean up the handling of the lower / higher parts of long values on 32 bit platforms . ( )
2008 - 06 - 01 Andreas Nahr < ClassDevelopment @ A - SoftTech . com >
- ThemeWin32Classic . cs : more date time picker painting fixes
2002 - 12 - 02 Dietmar Maurer < dietmar @ ximian . com >
Removed a cwl .
[ mkbundle ] Fix the argument order to put the entry point on argv [ 1 ] ( )
2004 - 12 - 20 Sebastien Pouliot < sebastien @ ximian . com >
2002 - 02 - 14 Jeffrey Stedfast < fejj @ ximian . com >
throw on protocol version mismatch
2002 - 03 - 19 Miguel de Icaza < miguel @ ximian . com >
Minor cleanup split out and expanded from another larger change . ( )
New test .
2007 - 05 - 22 Atsushi Enomoto < atsushi @ ximian . com >
2009 - 03 - 30 Carlos Alberto Cortez < calberto . cortez @ gmail . com >
Use i686 - pc - mingw32 - gcc as the mkbundle compiler on windows .
2005 - 09 - 29 Gonzalo Paniagua Javier < gonzalo @ ximian . com >
Handle cyclic interface definitions .
2007 - 03 - 28 Sebastien Pouliot < sebastien @ ximian . com >
in utils :
2008 - 12 - 09 Atsushi Enomoto < atsushi @ ximian . com >
* DataGrid . cs : When we get notified by a change in the position of our
[ exceptions ] continue program execution when mapping of stack overflow guard page fails ( )
Use faster versions of string comparers
2005 - 06 - 01 Sebastien Pouliot < sebastien @ ximian . com >
* tramp - s390 . c : Fix s390 build ( add missing pointer declarations
ucm2cp : Remove the deprecated ' # if NET_2_0 ' fence
2004 - 09 - 21 Geoff Norton < gnorton @ customerdna . com >
2005 - 12 - 08 Sebastien Pouliot < sebastien @ ximian . com >
* SerializationCodeGenerator . cs ( GenerateContract ) : Handle
2001 - 08 - 03 Miguel de Icaza < miguel @ ximian . com >
* dis - cil . c , dump . c , get . c , get . h , main . c : Replace several
2009 - 11 - 07 Rodrigo Kumpera < rkumpera @ novell . com >
Fix Android build . MINI_THREAD_VAR_OFFSET needs to be defined as MONO_THREAD_VAR_OFFSET ( which returns - 1 ) .
2007 - 10 - 31 Martin Baulig < martin @ ximian . com >
* CodeGeneratorFromTypeTest . cs : Added PrivateImplementationType tests
Merge pull request from ermshiperete / master
2009 - 03 - 29 Zoltan Varga < vargaz @ gmail . com >
Backport ReaderWriterLockSlim for MEF
2002 - 07 - 19 Martin Baulig < martin @ gnome . org >
2007 - 09 - 19 Martin Baulig < martin @ ximian . com >
2007 - 01 - 15 Chris Toshok < toshok @ ximian . com >
Fix NullReferenceException when reader fails
2010 - 01 - 03 Zoltan Varga < vargaz @ gmail . com >
Implement accessibility check for protected members via type parameter qualifier
ResXDataNode test refactoring , few more member tests .
* Additions for project save / load / edit
2006 - 10 - 17 Chris Toshok < toshok @ ximian . com >
Close WebResponse to free resources
Pass 1 of removing NET_2_0 from winforms
file is not in Wine include folder
* ComboBox . cs : On 2 . 0 , DropDownWidth throws ArgumentOutOfRangeException
Added counter mode in InternalGetChars .
2002 - 03 - 06 Martin Baulig < martin @ gnome . org >
2005 - 01 - 26 Marek Safar < marek . safar @ seznam . cz >
2007 - 11 - 04 Miguel de Icaza < miguel @ novell . com >
Add JIT counters to the local set of counters we support .
Will include the Windows version instead
[ btls ] Remove the statically linked btls code , the dynamic version should work on all platforms . ( )
Before combining two lines for a delete , we need
2010 - 05 - 05 Marek Safar < marek . safar @ gmail . com >
2008 - 11 - 24 Jonathan Pobst < monkey @ jpobst . com >
[ threads ] Fix leak of gchandle to MonoInternalThread
2005 - 04 - 13 Marek Safar < marek . safar @ seznam . cz >
Added RemoteActivator . cs
Fri Dec 7 22 : 08 : 23 CET 2007 Paolo Molaro < lupus @ ximian . com >
Stop warning about winsock2 . h being included before windows . h
typo
2002 - 04 - 28 Nick Drochak < ndrochak @ gol . com >
2005 - 09 - 10 Gonzalo Paniagua Javier < gonzalo @ ximian . com >
[ sdks ] Calculate sha1 of sdks archive and upload it
2005 - 02 - 02 Atsushi Enomoto < atsushi @ ximian . com >
[ wasm ] Update WASM - REQUESTS . md .
* CrossAppDomainChannel . cs : Fixed line endings . Set eol - style to native .
2005 - 04 - 19 Zoltan Varga < vargaz @ freemail . hu >
- Implemented ClientSize property
2005 - 10 - 04 Peter Dennis Bartok < pbartok @ novell . com >
fix csproj too .
[ Linker ] properly read symbols
Enable support for OP_MEMORY_BARRIER on llvm + arm , newer llvm versions support it .
* expression . cs :
exception fixes - unfortunately this does not work with mcs compiled binaries
2006 - 01 - 21 Martin Baulig < martin @ ximian . com >
2002 - 08 - 03 Gonzalo Paniagua Javier < gonzalo @ ximian . com >
2009 - 12 - 10 Rodrigo Kumpera < rkumpera @ novell . com >
2003 - 07 - 06 Andreas Nahr < ClassDevelopment @ A - SoftTech . com >
FlushetyFlushage
Return null instead of catching Activator exception ( and return null ) . What was a corner case before is now very common when the linker is used ( e . g . monotouch , m4a ) . Reported in assistly
tools / sqlmetal / AssemblyInfo . cs : Make internals visible to the sqlmetal testsuite .
[ sgen ] Avoid popping the entire finalizer queues if finalizers are suspended
* daemon - messages . c : Retry if the communication with the daemon is
2008 - 05 - 24 George Giolfan < georgegiolfan @ yahoo . com >
Fix wrong condition
Potential cygwin make issues fix
* Bitmap . jvm . cs : ArgumentException on unknown format from stream
2002 - 12 - 24 Atsushi Enomoto < ginga @ kit . hi - ho . ne . jp >
2007 - 06 - 13 Atsushi Enomoto < atsushi @ ximian . com >
update man page
New test
[ xbuild ] Fix dependency in C # targets
* UnixIOException . cs : Add ( string ) and ( string , Exception ) constructors to
Add cast to ( ( align ) - 1 ) for Alignment macros ( )
Fixed for Big Endian machines .
[ jit ] Trucate the shift amount in the long shift emulation functions to better mirror the behavior of the processor shift instructions .
merge 99208 : 99244
[ asp . net , routing ] Fix for bug . Don ' t match all - literal routes for URLs with excessive segments .
2004 - 06 - 18 Atsushi Enomoto < atsushi @ ximian . com >
* SqlDateTime . cs : Changed argument names to match MS . Code formatting .
Add logic to object array typecheck to handle arrays of unmanaged pointers ( )
[ readme ] Clarify - - enable - parallel - mark description
[ Web ] Fix obvious typo in DELETE FROM WHERE statement .
2002 - 10 - 15 Daniel Morgan < danmorg @ sc . rr . com >
2008 - 05 - 24 George Giolfan < georgegiolfan @ yahoo . com >
[ jit ] Fix a regression introduced by 00a28a7dfd44a10c3bc4c3b57449c93aa5124092 which disabled liveness2 for most methods .
fixed signtures
use the visibility hack for smcs
svn path = / branches / martin / anonymous - methods2 / mcs / ; revision = 68136
2005 - 12 - 01 Atsushi Enomoto < atsushi @ ximian . com >
Updated mdoc to remove < AssemblyVersions > tag
2002 - 03 - 20 Martin Baulig < martin @ gnome . org >
Debugging support in mcs .
Location of anon - SVN has moved
Revert ordering of compiler tests
2009 - 06 - 26 Atsushi Enomoto < atsushi @ ximian . com >
2005 - 10 - 18 Geoff Norton < gnorton @ customerdna . com >
[ mono - symbolicate ] Fixed wrong Directory . Exists check .
[ symbolicate ] Switch to Cecil backend
[ llvm ] Fix the computation of size of gshared instances in another place .
more RegionInfo work
2005 - 08 - 09 Gonzalo Paniagua Javier < gonzalo @ ximian . com >
2006 - 02 - 19 Zoltan Varga < vargaz @ gmail . com >
2010 - 05 - 07 Mark Probst < mark . probst @ gmail . com >
More flushage
2003 - 06 - 09 Ville Palo < vi64pa @ kolumbus . fi >
* ChannelServices . cs : Removed some TODOs . Implemented AsyncDispatchMessage .
2005 - 05 - 26 Ben Maurer < bmaurer @ ximian . com >
2007 - 11 - 08 Jb Evain < jbevain @ novell . com >
Code style
Fix a warning .
Test case for bugs and .
fix grammar in an exception
[ io - layer ] Remove another usage of getdtablesize ( ) .
2004 - 08 - 30 Zoltan Varga < vargaz @ freemail . hu >
2002 - 03 - 20 Martin Baulig < martin @ gnome . org >
Replace tfm based constants with SDK ones ( dotnet / corefx )
configure . in : version bump for 1 . 1 . 16
Merge pull request from LogosBible / locale - changes
Fix typo
Cosmetic changes .
2005 - 06 - 24 Zoltan Varga < vargaz @ freemail . hu >
2006 - 11 - 06 Alexander Olk < alex . olk @ googlemail . com >
2010 - 03 - 22 Sebastien Pouliot < sebastien @ ximian . com >
2008 - 11 - 06 Alan McGovern < amcgovern @ novell . com >
Merge pull request from madewokherd / mingw - build
[ threads ] Add MonoError arg to create_thread
2004 - 09 - 23 Zoltan Varga < vargaz @ freemail . hu >
added missing members
2009 - 04 - 02 Marek Habersack < mhabersack @ novell . com >
Fixed a bug w . r . t operations involving enum and other types .
- Removed usage of Rectangle for drawing . Miguel pointed out it ' s faster
remove confusing debug spew
2007 - 02 - 08 Jonathan Chambers < joncham @ gmail . com >
[ System . Data ] use PAL_gssapi . c for SSPI ( )
2006 - 02 - 14 Atsushi Enomoto < atsushi @ ximian . com >
Added parameter UriParser to UriParseComponent .
Stunts to rename cs111 . cs to cs0111 . cs ; - )
2005 - 11 - 30 Atsushi Enomoto < atsushi @ ximian . com >
Remove redundant field
2005 - 01 - 28 Sebastien Pouliot < sebastien @ ximian . com >
[ System . Runtime . Serialization ] Work around ` specified cast is not valid ` error ( )
[ build ] Exclude generated . g . cs files from " make dist "
* ListTest . cs ( TestGetRange ) : Add missing ' ; ' . Actually compare
Tue Jan 21 17 : 29 : 53 CET 2003 Paolo Molaro < lupus @ ximian . com >
Remove Collection dependency from ReadOnlyCollection
Revert my last change to a submodule . ALL MY LOVE TO THOSE THAT INVENTED GIT SUBMODULES . < 3
Differentiate between IPostBackEventHandler that was registered by the user , and one that was given as post parameter , so that in case ther post data may trigger both IPostBackEventHandler and IPostBackDataHandler , only the IPostBackDataHandler is actually triggered
2007 - 11 - 23 Zoltan Varga < vargaz @ gmail . com >
Fix building sgen test suite .
[ io - layer ] Remove unused _wapi_sem_id
2003 - 11 - 15 Todd Berman < tberman @ gentoo . org >
2009 - 11 - 17 Zoltan Varga < vargaz @ gmail . com >
2009 - 04 - 13 Marek Safar < marek . safar @ gmail . com >
2006 - 06 - 22 Marek Sieradzki < marek . sieradzki @ gmail . com >
* TreeView . cs : Use the horizontal scrollbars height not width when
merge r67228 - r67235 , r67237 , r67251 and r67256 - 67259 to trunk ( they are
Filename fix
[ corlib ] Fix GetFrames_AsyncCalls test not to block
[ runtime ] Handle generic methods in interfaces in imt thunks correctly . Fixes crashes in the System . ServiceModel tests .
2007 - 08 - 13 Jb Evain < jbevain @ novell . com >
make delegate and method private
Rework the gsharedvt code to add support for vtypes with arbitrary size .
gdi32 . dll redirection apply to non - windows only
2006 - 08 - 23 Gonzalo Paniagua Javier < gonzalo @ ximian . com >
Fix
* PEAPI . cs : Allow hash algorithms to be any integer .
* Constraint . cs : marked methods internal to fix public API
2003 - 02 - 19 Ville Palo < vi64pa @ kolumbus . fi >
More NET_2_0 define removal
Small change that escaped last commit making the build to fail .
[ sgen ] Explicitly indicate if we run jobs on the worker thread
2008 - 02 - 10 Sebastien Pouliot < sebastien @ ximian . com >
2004 - 06 - 25 Atsushi Enomoto < atsushi @ ximian . com >
disable Win32 registry dependencies in XmlReaderSettings on MOBILE builds .
2004 - 09 - 07 Atsushi Enomoto < atsushi @ ximian . com >
Add flags for Release config ( enable optimization ) .
2002 - 04 - 05 Dietmar Maurer < dietmar @ ximian . com >
2005 - 06 - 28 Sureshkumar T < tsureshkumar @ novell . com >
2005 - 04 - 07 Gonzalo Paniagua Javier < gonzalo @ ximian . com >
bug fix in Enum : : Equals
New test .
* Uri . cs ( InternalEquals ) : New helper .
2006 - 11 - 18 Miguel de Icaza < miguel @ novell . com >
2004 - 07 - 21 Atsushi Enomoto < atsushi @ ximian . com >
2001 - 12 - 06 Miguel de Icaza < miguel @ ximian . com >
[ ci ] Fix typo in sdks - archive . groovy
Bump Roslyn
2010 - 05 - 11 Miguel de Icaza < miguel @ novell . com >
Fix
Fix few unit tests
2006 - 01 - 15 Jordi Mas i Hernandez < jordimash @ gmail . com >
Commit
2009 - 05 - 25 Carlos Alberto Cortez < calberto . cortez @ gmail . com >
Initialize klass - > has_finalizer lazily . This allows us to remove the find_method_in_metadata ( ) optimization without a perf cost which is broken in the presence of overrides . Fixes .
2005 - 04 - 26 Sebastien Pouliot < sebastien @ ximian . com >
2004 - 02 - 26 Sebastien Pouliot < sebastien @ ximian . com >
2006 - 01 - 09 Chris Toshok < toshok @ ximian . com >
2005 - 01 - 17 Atsushi Enomoto < atsushi @ ximian . com >
Merge pull request from lateralusX / lateralusX / external - direct - icalls - symbol - name
2003 - 12 - 24 Atsushi Enomoto < atsushi @ ximian . com >
[ netcore ] Install and use GNU sed on Mac
2005 - 05 - 09 Sebastien Pouliot < sebastien @ ximian . com >
2009 - 11 - 17 Atsushi Enomoto < atsushi @ ximian . com >
A few memory saving updates from my tree
initialize ReaderQuotas , fixes bugzilla
[ mini ] Remove some unstable exports that are no longer used by the profiler .
2009 - 10 - 07 Atsushi Enomoto < atsushi @ ximian . com >
[ monodoc ] Change new API for Nodes to be IList for future - proofing
2004 - 05 - 27 Umadevi S < sumadevi @ novell . com >
[ wasm ] Add support for pinvokes in user assemblies . ( )
[ mcs ] Add string interpolation conversions
2005 - 06 - 08 Zoltan Varga < vargaz @ freemail . hu >
* make - map . cs : Process the Mono . Posix namespace as well as Mono . Unix ; add
2009 - 11 - 07 Rodrigo Kumpera < rkumpera @ novell . com >
Add ChangeLog
2002 - 10 - 07 Gaurav Vaish < gvaish_mono @ lycos . com >
2007 - 10 - 09 Rodrigo Kumpera < rkumpera @ novell . com >
2004 - 05 - 11 Atsushi Enomoto < atsushi @ ximian . com >
2004 - 07 - 08 Sebastien Pouliot < sebastien @ ximian . com >
Copy
* list : added System . Web . Services . Protocols . HttpSoapWebServiceHandler . cs and
2005 - 11 - 23 Chris Toshok < toshok @ ximian . com >
Updating FormView Test
fix a mistake in the previous patch
2008 - 01 - 14 Sephane Delcroix < sdelcroix @ novell . com >
Update embedded api
[ mcs ] Use $ ( PLATFORM_PATH_SEPARATOR ) instead of ' : ' for MONO_PATH to fix Windows tests
Filter out more FrameworkEventSource
mysql exception classes forgot to be added cvs
2005 - 09 - 24 Gonzalo Paniagua Javier < gonzalo @ ximian . com >
Added System . Web . J2EE / PageMapper . cs and fixed dependecies in J2EE config .
Changed compiler directive according to documentation
2004 - 09 - 06 Atsushi Enomoto < atsushi @ ximian . com >
2007 - 02 - 13 Igor Zelmanovich < igorz @ mainsoft . com >
2008 - 06 - 17 Marek Habersack < mhabersack @ novell . com >
More removal of old defines
2010 - 04 - 14 Atsushi Enomoto < atsushi @ ximian . com >
[ ios sdk ] Package resource assemblies into app bundle ( )
2003 - 10 - 07 Bernie Solomon < bernard @ ugsolutions . com >
fix the 2 . 0 build
[ mini ] Lower atomic load / store instruction sizes on ARM .
2007 - 09 - 05 Marek Habersack < mhabersack @ novell . com >
Various fixes for bugs reported by Michael : erratic behavior at end of line , problems with yanking
Extract a macro to loop over all threads .
2007 - 03 - 15 Jonathan Chambers < joncham @ gmail . com >
[ asp . net ] Optimize memory usage a bit . String hashes are cached in thread - local storage .
Oops , fix
2007 - 03 - 22 Igor Zelmanovich < igorz @ mainsoft . com >
2006 - 04 - 06 Alexander Olk < alex . olk @ googlemail . com >
updated build with latest mcs in cvs
Tue Mar 1 18 : 35 : 27 CET 2005 Paolo Molaro < lupus @ ximian . com >
2007 - 05 - 30 Nagappan A < anagappan @ novell . com >
[ cxx ] Fix enum / int mixup in new code . ( )
add parantheses for MONO_REGISTRY_PATH
New test .
Working on Attribute methods for XmlTextWriter .
Add managed implementation of GetSystemMaxDBCSCharSize ( )
Testcases for Array Creation expressions - by Sudharsan V
[ test ] Rename test
2005 - 03 - 28 Sebastien Pouliot < sebastien @ ximian . com >
[ mcs ] Extend emit preparation to binary expressions . Fixes
[ sgen ] Remove superfluous function indirection .
Doc for AtomicBoolean ( Value )
* XplatUIStructs . cs : Fix a couple wrong virtual keycodes .
2002 - 10 - 29 Gaurav Vaish < gvaish_mono AT lycos . com >
2005 - 12 - 02 Robert Jordan < robertj @ gmx . net >
2006 - 12 - 11 Rolf Bjarne Kvinge < RKvinge @ novell . com >
2005 - 10 - 08 Zoltan Varga < vargaz @ freemail . hu >
2009 - 10 - 15 Marek Safar < marek . safar @ gmail . com >
[ msvc ] Update csproj files ( )
[ System . ServiceModel . Web ] Fix issue with types explicitly implementing IDictionary or IDictionary < , >
Reverted back ' UserProperties ' section in project file .
removed a hello world from the code .
2005 - 03 - 17 Martin Baulig < martin @ ximian . com >
2004 - 07 - 13 Peter Williams < peter @ newton . cx >
2004 - 08 - 30 Atsushi Enomoto < atsushi @ ximian . com >
Fall back to calling Java . Util . TimeZone . Default . ID if the persist . sys . timezone system
2003 - 06 - 29 Nick Drochak < ndrochak @ gol . com >
Sat Mar 16 19 : 11 : 47 CET 2002 Paolo Molaro < lupus @ ximian . com >
[ cxx ] Strengthen some types away from gpointer . ( )
[ sgen - bridge ] Fix forwarding .
- Added private method to get the Control object from the window handle
Documentation updates
2009 - 08 - 18 Rodrigo Kumpera < rkumpera @ novell . com >
Merge pull request from lambdageek / bug - 49721
2002 - 08 - 19 Ravi Pratap < ravi @ ximian . com >
fix a typo
2008 - 06 - 27 Zoltan Varga < vargaz @ gmail . com >
2009 - 09 - 04 Gonzalo Paniagua Javier < gonzalo @ novell . com >
[ interpreter ] add more regression tests to generics - interp . cs
2009 - 10 - 24 Zoltan Varga < vargaz @ gmail . com >
Adding ArrayListTest to AllTests
2003 - 01 - 26 Gonzalo Paniagua Javier < gonzalo @ ximian . com >
In System . Text . RegularExpressions :
2008 - 04 - 28 George Giolfan < georgegiolfan @ yahoo . com >
2005 - 07 - 20 Atsushi Enomoto < atsushi @ ximian . com >
Fix generation of wrong version .
2008 - 11 - 04 Zoltan Varga < vargaz @ gmail . com >
BigIngeter : new LeadingZeroCount helper method
Changed mono_path profile replacement to be case insensitive .
In System . Text . RegularExpressions :
Include MVC3 and aspnetwebstack assemblies pkg - config files
Tue Dec 5 19 : 19 : 34 CET 2006 Paolo Molaro < lupus @ ximian . com >
[ interpreter ] remove wrong error trap
More updates
Fix the bitcode build .
Bump Roslyn to 2 . 8 . 2
Update System . Data . Entity to handle new String overloads ( )
[ threads ] Enter GC Unsafe before pumping HP queue in unregister_thread ( )
Fix the handling of unresolved pinvoke methods in full - aot mode .
[ wcf ] Make sure context exists in the dictionary getting its associated wait handle .
Now it ' s working without any ugly hacks : - )
2008 - 09 - 08 Jonathan Pobst < monkey @ jpobst . com >
Handle OOM with string interning .
Adding basic codeview support on Windows x64 full AOT builds .
change from wndproc to events
2005 - 04 - 23 Atsushi Enomoto < atsushi @ ximian . com >
Give the minor copy functions different names under for each minor collector to make profilers ( Intruments ) happy .
Register two separate functions for mono_fconv_u4 and mono_fconv_u8
2005 - 09 - 20 Sureshkumar T < tsureshkumar @ novell . com >
2005 - 07 - 01 Lluis Sanchez Gual < lluis @ novell . com >
2008 - 11 - 20 Rodrigo Kumpera < rkumpera @ novell . com >
Updates .
Bump sdb protocol version in the runtime .
Enable aot - ing of mscorlib + gmcs on all architectures where AOT is supported .
Fix typos in " man sn " .
Handle more special keys in KeysConverter to ensure compatibility with designer generated resource files .
[ llvm ] Fix a few problems with longs on 32 bit platforms . Fixes .
2005 - 12 - 25 Daniel Morgan < danielmorgan @ verizon . net >
* HttpResponseTest . cs : Test_Response , TestResponse_Chunked marked as NotWorking in GH
In mcs and gmcs :
By : Yonik < yonik @ mainsoft . com >
[ corlib ] Clean up some warnings
[ runtime ] Don ' t crash when method / assembly with no code coverage does not have debug symbols
Revert WriteLine
2001 - 12 - 22 Miguel de Icaza < miguel @ ximian . com >
[ sdb ] Avoid placing implicit sequence points at offsets where the IL stack is empty when using symbol file based sequence points . Fixes .
2004 - 06 - 18 Umadevi S < sumadevi @ novell . com >
2004 - 04 - 02 Sebastien Pouliot < sebastien @ ximian . com >
[ acceptance - tests ] Append a suffix to the CoreCLR IL tests
2004 - 05 - 12 Atsushi Enomoto < atsushi @ ximian . com >
[ class ] Use constraints of gparam T2 in T1 . IsAssignableFrom ( T2 )
[ bcl ] Update Reference Source to . NET Framework 4 . 6 . 2
2009 - 08 - 14 Marek Habersack < mhabersack @ novell . com >
Add new test case .
Small hack to avoid the make - f makefile . gnu
2004 - 03 - 22 Zoltan Varga < vargaz @ freemail . hu >
Removes PNSE version of DispatchWrapper and keep only single version with ifdef
Documented the current version for the log profiler data format .
[ build ] Remove multi stage build from Mono . Security ( )
2005 - 08 - 29 Chris Toshok < toshok @ ximian . com >
In . :
2004 - 11 - 04 Ben Maurer < bmaurer @ ximian . com >
2003 - 01 - 26 Daniel Morgan < danmorg @ sc . rr . com >
Emit proper unwind info for the gsharedvt trampoline in full - aot mode . Fix gsharedvt support in mono_aot_plt_resolve ( ) . Add a ' no - instantiations ' AOT option to avoid the generation of generic instances , to allow better testing of the gsharedvt code .
[ System . Xml ] Fix sub class serialization on lists
2010 - 01 - 20 Atsushi Enomoto < atsushi @ ximian . com >
Fix 5d37fac7c8765253a2ae406329ebf360c8429387 yet again .
2008 - 07 - 22 Marek Safar < marek . safar @ gmail . com >
set eolstyle correctly
2010 - 04 - 09 Rodrigo Kumpera < rkumpera @ novell . com >
2009 - 02 - 02 Rodrigo Kumpera < rkumpera @ novell . com >
Add
Define a symbol for arch that have the jit ported .
* stdio . c : Add Mono_Posix_Stdlib_DumpFilePosition , which create a hex string
* * * * Merged r37687 - r37734 from MCS * * * *
2002 - 08 - 18 Rodrigo Moya < rodrigo @ ximian . com >
[ mini ] emit sequence point before branching off for tail call
2007 - 06 - 20 Juraj Skripsky < js @ hotfeet . ch >
2010 - 07 - 23 Marek Habersack < mhabersack @ novell . com >
[ ilasm ] fix the metadata version for v4 assemblies
2009 - 06 - 25 Atsushi Enomoto < atsushi @ ximian . com >
2004 - 04 - 15 Bernie Solomon < bernard @ ugsolutions . com >
2004 - 04 - 21 Marek Safar < marek . safar @ seznam . cz >
* EventLogTest . cs : new test , currently not enabled due to UnixRegistry
Add IsVariableBoundArray to TypeDelegator ( dotnet / coreclr )
2009 - 03 - 03 Rodrigo Kumpera < rkumpera @ novell . com >
oops
Treat the frame reg as precisely tracked .
[ jit ] Add an intrinsics for BitConverter . IsLittleEndian . ( )
2009 - 01 - 09 Marek Habersack < mhabersack @ novell . com >
[ sdb ] Fix the decoding of the next array when decoding sequence points .
[ mdoc - update - ecma - xml ] Sort libraries and types .
2004 - 08 - 23 Martin Baulig < martin @ ximian . com >
Bump corefx
2008 - 08 - 13 Marek Safar < marek . safar @ gmail . com >
Update version information in System . Private . CoreLib to mimic CoreCLR and make BenchmarkDotNet happy ( )
2002 - 10 - 01 Duncan Mak < duncan @ ximian . com >
2010 - 03 - 19 Atsushi Enomoto < atsushi @ ximian . com >
2003 - 06 - 16 Atsushi Enomoto < ginga @ kit . hi - ho . ne . jp >
[ runtime ] Specialize run - jenkins script for mobile_static when it runs
2007 - 08 - 24 Jonathan Pobst < monkey @ jpobst . com >
2002 - 05 - 08 Tim Coleman < tim @ timcoleman . com >
2003 - 11 - 04 cesar lopez nataren < cesar @ ciencias . unam . mx >
2008 - 10 - 11 Zoltan Varga < vargaz @ gmail . com >
Small fix .
Add mono_arch_get_nullified_class_init_trampoline
Re - ordered the * IsValid check in SiteIdentityPermissions . We were throwing an IndexOutOfRangeException
2009 - 03 - 16 Carlos Alberto Cortez < calberto . cortez @ gmail . com >
2008 - 01 - 24 Marek Safar < marek . safar @ gmail . com >
Bump bockbuild to get GTK NSPasteboardType change . ( )
2007 - 09 - 06 Jonathan Pobst < monkey @ jpobst . com >
[ System . Security ] CoreFX import for SubjectIdentifierOrKey type .
2004 - 06 - 11 Atsushi Enomoto < atsushi @ ximian . com >
Move more cardtable specific code to its file .
2007 - 05 - 31 Atsushi Enomoto < atsushi @ ximian . com >
* Calendar . cs : Match MS postback data . This allows sites that
2006 - 08 - 17 Zoltan Varga < vargaz @ gmail . com >
Add 4 . 5 profile support to xbuild
[ lldb ] update expression in script
[ System . ComponentModel . DataAnnotations ] Implemented EmailAddressAttribute . IsValid ( )
2004 - 02 - 17 Atsushi Enomoto < atsushi @ ximian . com >
* DataSet . cs : Convert to a partial class and sequester NET_2_0
Fixes build
* image . h ( MONO_IMAGE_IS_CORLIB ) : new macro to check root image
[ coop ] Define mono_threads_is_blocking_transition_enabled in terms of suspend policy
* ilasm . exe . sources : Add FileRef . cs
2002 - 02 - 14 Jeffrey Stedfast < fejj @ ximian . com >
2002 - 05 - 15 Radek Doulik < rodo @ ximian . com >
[ corlib ] Improve 85aaf9b07485d5253de4c142985e32b00ad19347 change
2002 - 02 - 26 Duncan Mak < duncan @ ximian . com >
Add test for bug
[ crypto ] Key size 128 bytes for HMACSHA384
Add test for g_utf8_to_utf16_with_nuls .
Fail LLVM compilation when using sgen + OP_STOREV_MEMBASE with structs with reference fields . Fixes .
* DataGridViewCell . cs : EditType : returns
2008 - 04 - 29 George Giolfan < georgegiolfan @ yahoo . com >
return null for nested static types in XamlSchemaContext . GetXamlNamespace ( ) .
Remove file
[ System ] Fixes SmtpClientTest . Deliver_Async
Flush
add Precisios / Scale implementations to comply . net signatures
2008 - 09 - 05 Jonathan Pobst < monkey @ jpobst . com >
2009 - 04 - 27 Atsushi Enomoto < atsushi @ ximian . com >
Some Sparc realted fixes . Doesn ' t compile yet , but getting closer .
Fix verifier test suite to build on OSX as well
[ reflection ] ModuleBuilder , TypeBuilder escaped type identifiers patch
[ runtime ] Only add Mono . Dynamic . Interpreter to mobile_static ( )
[ sgen ] Shorter IR for inlined write barrier .
Fix make distcheck .
Re - enabled CS1612 test .
2008 - 07 - 06 Andreas Nahr < ClassDevelopment @ A - SoftTech . com >
New test .
Delay expensive inflator construction for type parameter constraints
[ System ] : Add HttpListenerBasicIdentity . platformnotsupported . cs for watch . ( )
* ReflectionEventDescriptor . cs : Bind handlers to the actual event so that
[ interp ] Remove unconditional branches to next instruction ( )
Merge pull request from knocte / master
2008 - 05 - 02 Jonathan Pobst < monkey @ jpobst . com >
2002 - 10 - 06 Martin Baulig < martin @ gnome . org >
Adding SqlTypes tests to the project file
Add a . gitignore file to support / minizip / to ignore * . la * . lo files .
[ sdb ] Move the checking of the various debugger attributes to one function to make it easier to add support for DebuggerStepperBoundaryAttribute .
Flush
2003 - 04 - 28 Zoltan Varga < vargaz @ freemail . hu >
2007 - 09 - 02 Igor Zelmanovich < igorz @ mainsoft . com >
2010 - 04 - 29 Jérémie Laval < jeremie . laval @ gmail . com >
Move target for creating monolite package to Mono
2003 - 09 - 30 Zoltan Varga < vargaz @ freemail . hu >
2007 - 01 - 23 Chris Toshok < toshok @ ximian . com >
Changed mini - exceptions functions to get IL offsets from seq point . mono_walk_stack_full and ves_icall_get_trace are now using find_prev_seq_point_for_native_offset to retrieve IL offsets .
[ Debugger ] Fix crash when there is a generic struct with a field that is an enumerator . ( )
sorted out BuildManager build logic up to BuildSubmission . Execute ( ) .
2004 - 11 - 17 Martin Baulig < martin @ ximian . com >
2004 - 11 - 25 Zoltan Varga < vargaz @ freemail . hu >
[ sgen ] Clear nursery fragments before whole heap check .
Revert changes in SWF . csproj
' System . Runtime . Serialization . StreamingContext is a structure and can not be compared with null . Fix compilation error
Mon Mar 10 11 : 59 : 34 CET 2008 Paolo Molaro < lupus @ ximian . com >
2010 - 03 - 05 Carlos Alberto Cortez < calberto . cortez @ gmail . com >
2010 - 03 - 10 Atsushi Enomoto < atsushi @ ximian . com >
[ Mono . Options ] Add ` CommandSet . GetCompletions ( ) ` ( )
2005 - 08 - 31 Sebastien Pouliot < sebastien @ ximian . com >
Merge pull request from lambdageek / monoerror - reflection - bubble
2006 - 08 - 31 Zoltan Varga < vargaz @ gmail . com >
2003 - 10 - 22 Dick Porter < dick @ ximian . com >
2010 - 03 - 08 Carlos Alberto Cortez < calberto . cortez @ gmail . com >
TARGET_JVM : correctly handle Portal ' s ' GetBack '
[ crypto ] check key length in test
[ corlib ] Ignore metadata on stacktrace tests
Fix IDE0008 ( use explicit type instead of var )
2010 - 01 - 26 Atsushi Enomoto < atsushi @ ximian . com >
2002 - 08 - 01 Miguel de Icaza < miguel @ ximian . com >
2005 - 06 - 06 Marek Safar < marek . safar @ seznam . cz >
make this 2 . 0 only
2009 - 03 - 16 Rodrigo Kumpera < rkumpera @ novell . com >
* mddump . cs : Comment out some stuff that does not need to be seen .
Tue Oct 16 17 : 22 : 15 CEST 2007 Paolo Molaro < lupus @ ximian . com >
ChangeLog : Updated ChangeLog .
* ActivationServices . cs : Added method for creating a proxy from a list
[ acceptance - tests ] Move roslyn build dependencies to download server
2006 - 08 - 04 Gonzalo Paniagua Javier < gonzalo @ ximian . com >
[ mono ] Remove unused variable .
Document error
Tue Jan 16 15 : 32 : 53 CET 2007 Paolo Molaro < lupus @ ximian . com >
Disable another runtime failing test
Merge pull request from akoeplinger / mono - posix - test - fix
Merge pull request from lewurm / finalizer_thread
2007 - 06 - 13 Marek Safar < marek . safar @ gmail . com >
* VBCodeProviderTest . cs : Use mono to execute assembly .
2004 - 11 - 05 Zoltan Varga < vargaz @ freemail . hu >
2004 - 04 - 26 David Waite < mass @ akuma . org >
2006 - 08 - 31 Zoltan Varga < vargaz @ gmail . com >
2007 - 01 - 15 Carlos Alberto Cortez < calberto . cortez @ gmail . com >
- Fixed signature
[ meta ] Fix loader printf mistake and outdated info in Makefile ( )
BGR to RGB function
Additional constructor preservations
Fix typo .
Ignore ExceptionDispatchInfoTest : Throw ( ) until it is fixed ( ) .
2001 - 09 - 23 Dick Porter < dick @ ximian . com >
Corrections to * Trim sent by Daniel Campos
Rearrange some code to improve consistency
2010 - 01 - 15 Zoltan Varga < vargaz @ gmail . com >
NS2 . 1 : System . Security . Cryptography changes ( )
Bump msbuild to track xplat - master ( )
[ runtime ] Fix parse_optimizations ( ) so the optimization names don ' t … ( )
* Binding . cs , Import . cs , Message . cs , MessageBinding . cs , MessagePart . cs ,
[ Cleanup ] Cleanup per vargaz comments .
2001 - 10 - 26 Miguel de Icaza < miguel @ ximian . com >
* runtime / Makefile . am ( SUPPORT_FILES ) : New . List of dependencies from . . .
revert simplification
Avoid computing GC maps if precise stack marking is disabled .
2001 - 12 - 09 Miguel de Icaza < miguel @ ximian . com >
2004 - 04 - 06 Andreas Nahr < ClassDevelopment @ A - SoftTech . com >
[ runtime ] Fix a loader leak in the aot compiler .
2008 - 08 - 21 Sebastien Pouliot < sebastien @ ximian . com >
2005 - 06 - 10 Martin Baulig < martin @ ximian . com >
fixed test
2003 - 10 - 04 Sebastien Pouliot < spouliot @ videotron . ca >
[ interp ] use libmonoldflags ( to enforce - no - undefined )
Zeroing is not necessary , snprintf ( ) always null - terminates in this case
2006 - 03 - 31 Alexander Olk < alex . olk @ googlemail . com >
2006 - 04 - 18 Gonzalo Paniagua Javier < gonzalo @ ximian . com >
[ jit ] Use compile - time conditionals in a few places .
awaiterbuilder cannot be readonly , it causes unexpected behaviour on . net
2010 - 09 - 16 Veerapuram Varadhan < vvaradhan @ novell . com >
Add and apply nullable attributes ( dotnet / coreclr )
Also update this
Remove debug comment
Fix mcs sdk default
Fixes a warning in System . Web . . Services . dll
Bump api - snapshot
* OleDbDataReader . cs : Removed bogus MonoTODO .
2008 - 03 - 25 Zoltan Varga < vargaz @ gmail . com >
[ asp . net ] Added code to handle local resources copying in the test suite
2005 - 03 - 29 Zoltan Varga < vargaz @ freemail . hu >
2003 - 08 - 18 Ben Maurer < bmaurer @ users . sourceforge . net >
* test - 695 . cs :
Fri Jun 7 19 : 39 : 43 CEST 2002 Paolo Molaro < lupus @ ximian . com >
2005 - 02 - 20 Zoltan Varga < vargaz @ freemail . hu >
[ netcore ] Add DARC subscriptions for Microsoft . Private . CoreFx . NETCoreApp
2007 - 10 - 24 Jonathan Pobst < monkey @ jpobst . com >
Fixed the build
* CodeGen . cs : Add code buffer for il instructions when a new method is added .
2009 - 01 - 23 Ivan N . Zlatev < contact @ i - nz . net >
Merge pull request from lambdageek / dev / checked - imageset - subset
2009 - 10 - 16 Miguel de Icaza < miguel @ novell . com >
[ threadpool - io ] Disable AIO by default
Applied patch from Simon Guindon to fix bug
Implement support for OP_LOADR4_MEMBASE / STORER4_MEMBASE with big offsets on ARM . Fixes .
2007 - 05 - 01 Zoltan Varga < vargaz @ gmail . com >
Fix the 1 . 0 dropping
[ System . Core ] Implemented PointerOffset
2003 - 07 - 11 Zoltan Varga < vargaz @ freemail . hu >
2004 - 06 - 22 Atsushi Enomoto < atsushi @ ximian . com >
* ThemeNice . cs : Use XplatUI . RunningOnUnix instead of checking
svn path = / trunk / mcs / ; revision = 53107
Merge pull request from akoeplinger / mingw - fix
Enable LLVM compilation of runtime invoke wrappers .
2004 - 07 - 05 Dick Porter < dick @ ximian . com >
Patch .
ChangeLog : Update ChangeLog .
Move some CustomAttribute related types to shared ( dotnet / coreclr )
Eglib integrate ( )
2010 - 02 - 03 Sebastien Pouliot < sebastien @ ximian . com >
[ interp ] fix stack overflow on tail recursion
Change # if line position in BarrierPostPhaseException to be below licence text
2005 - 11 - 15 Zoltan Varga < vargaz @ gmail . com >
[ coop ] Shorten the blocking section in threadpool cleanup .
Fixed unreachable methods in mono - thread - windows to use g_assert_not_reached .
Change help format
Fix warnings after API changes .
2009 - 02 - 07 Zoltan Varga < vargaz @ gmail . com >
2008 - 09 - 29 Marek Habersack < mhabersack @ novell . com >
Update
2007 - 11 - 07 Marek Safar < marek . safar @ gmail . com >
[ monkeydoc ] Robustify EcmaDesc equality checking and fix unit tests
Partially implement ExceptionDispatchInfo
This may help with threads in FreeBSD .
2005 - 08 - 12 Martin Baulig < martin @ ximian . com >
2003 - 11 - 27 Ben Maurer < bmaurer @ users . sourceforge . net >
2004 - 06 - 03 Atsushi Enomoto < atsushi @ ximian . com >
* driver . cs : More fiddling with paths .
2010 - 07 - 23 Zoltan Varga < vargaz @ gmail . com >
2005 - 02 - 10 Zoltan Varga < vargaz @ freemail . hu >
implement a relaxed mode of CoreCLR for coreclr users that do not require to be like moonlight 100 %
2008 - 06 - 27 Andreas Nahr < ClassDevelopment @ A - SoftTech . com >
2002 - 10 - 04 Miguel de Icaza < miguel @ ximian . com >
* DataGrid . cs : Create columns when the binding context has been
More inline docs
* SqlConnectionTest . cs : Added tests for BeginTransaction and GetSchema
2004 - 12 - 02 Miguel de Icaza < miguel @ ximian . com >
Merge pull request from brendanzagaeski / patch - 1
[ arm64 ] Allow v8 . . v15 in unwind info on arm64 . Fixes part of .
[ mcs ] Implements dictionary initializers on pointer types
[ runtime ] mark mono_domain_unload external only
Try Gaurav ' s
refactoring
2004 - 07 - 21 Bernie Solomon < bernard @ ugsolutions . com >
2009 - 11 - 03 Rodrigo Kumpera < rkumpera @ novell . com >
Apply patch from Paolo :
Add prebuilt resources file .
2008 - 04 - 15 Jonathan Chambers < joncham @ gmail . com >
* TabPage . cs : Add a new method so tab pages can force the tab
[ sre ] Add MonoAssemblyBuilderAccess enum in native code
config - default . make : Corrected default CODEPAGE comment to ISO - 8859 - 1 ( Latin 1 )
Move OperationStatus into shared source ( dotnet / coreclr )
Merge pull request from directhex / master
[ mono - threads ] Fix win32 to posix priority conversion
2009 - 02 - 18 Bill Holmes < billholmes54 @ gmail . com >
2004 - 11 - 07 Atsushi Enomoto < atsushi @ ximian . com >
[ wasm ] Fix master and add sample . ( )
Fix a warning .
Merge pull request from DavidKarlas / sdbTypesCaching
Wed Aug 22 18 : 35 : 12 CEST 2007 Paolo Molaro < lupus @ ximian . com >
2010 - 03 - 31 Miguel de Icaza < miguel @ novell . com >
changelog entry
Unify EH callbacks .
[ corlib ] Deconstruct api from corert
2007 - 05 - 09 Sebastien Pouliot < sebastien @ ximian . com >
2005 - 07 - 02 Martin Baulig < martin @ ximian . com >
2006 - 12 - 10 Marek Safar < marek . safar @ gmail . com >
Add new test
[ msvc ] Update csproj files ( )
2003 - 03 - 11 : Jean - Marc ANDRE < jean - marc . andre @ polymtl . ca >
update nunit options that break upgrade to 2 . 4 .
2008 - 01 - 15 Atsushi Enomoto < atsushi @ ximian . com >
[ runtime ] Allow getting thread name , managed id from embedding API
2007 - 07 - 20 Jb Evain < jbevain @ novell . com >
2008 - 10 - 01 Zoltan Varga < vargaz @ gmail . com >
2007 - 08 - 19 Andreia Gaita < avidigal @ novell . com >
* XsdValidatingReader . cs : In MoveToAttribute ( name ) , if " name " is a default
2004 - 06 - 06 Atsushi Enomoto < atsushi @ ximian . com >
[ sgen ] Don ' t report that we need a major collection while one is in progress .
add ISafeSerializationData
In System . Web . UI :
[ debugger ] Fix suspend_policy that will be send to debugger - libs ( )
libgc : Use GCC atomics on ARM .
* BitConverter . cs : Fixed exceptions to match MS . Fixed invalid use
[ System . Web ] Fixes SkinID property declaration
Copy
Update MonoTaskExtensions doc
[ wasm ] [ sdk ] Fixes to the sdk packages ( )
In Dns . cs implemented proper exception handling when trying to resolve not existing address ( under ' TARGET_JVM ' )
2010 - 04 - 14 Zoltan Varga < vargaz @ gmail . com >
fix typo .
2002 - 08 - 12 Tim Coleman < tim @ timcoleman . com >
New attribute and enum introduced in 2 . 0 beta2
* Color . cs : Fixed line endings . Set eol - style to native .
Merge pull request from directhex / master
* ComboBox . cs :
Update ` , gitignore ` ( )
Updated corlib version number .
[ corlib ] Tests TimeZoneInfo . ParseTZBuffer
[ API ] Added missing . Net 4 . 5 interface and dependent class
* RemotingProxy . cs : Use CallType property to check the type of
2005 - 06 - 04 Peter Bartok < pbartok @ novell . com >
Merge pull request from adbre / iss3333
2004 - 07 - 12 Massimiliano Mantione < massi @ ximian . com >
First set of licensing changes
[ threading ] [ jit ] Fix non coop osx build .
Remove unused ves_icall_System_Globalization_RegionInfo_construct_internal_region_from_lcid . ( )
Set vts tests environment variables
2006 - 10 - 27 Rolf Bjarne Kvinge < RKvinge @ novell . com >
* DataViewTest . cs : Add some tests for new methods .
Fixed g_build_path ( ) ( was failing a test )
[ runtime ] Enable fp elimination on winx64 .
2002 - 08 - 08 Martin Baulig < martin @ gnome . org >
[ llvmonly ] Skip initalization of methods if they only reference shared got entries .
Fix automake warning
Add a third AOT location , alongside ~ / . mono / aot - cache and the assembly dir
2007 - 10 - 23 Zoltan Varga < vargaz @ gmail . com >
NaCl x86 codegen fixes
Ignore private methods in batch import
Correctly handle nested generic scopes .
Initial sgen mips port .
* monoresgen . cs : Added support for 4 digit unicode escapes on 2 . 0
2006 - 01 - 22 Chris Toshok < toshok @ ximian . com >
Add test for bug ug
Update for Grasshopper 2 . 2
Very close , only need 3 more functions + replacing mono - hash with ours
2008 - 06 - 14 Zoltan Varga < vargaz @ gmail . com >
* PEAPI . cs : Fix pinvoke attributes , zero a methods RVA if it has
2007 - 09 - 11 Jonathan Pobst < monkey @ jpobst . com >
it is largely XHTMLified .
2006 - 07 - 13 Zoltan Varga < vargaz @ gmail . com >
[ sgen ] Add old forgotten oom tests to sgen oom suite
2006 - 05 - 12 Chris Toshok < toshok @ ximian . com >
2008 - 09 - 12 Jb Evain < jbevain @ novell . com >
2007 - 04 - 25 Atsushi Enomoto < atsushi @ ximian . com >
Correction in test case
new test
2004 - 05 - 10 Atsushi Enomoto < atsushi @ ximian . com >
[ monodoc ] Add a test for ba222d4eb939ef097dd2b77b1fea5c3b8a60b310
[ Decimal ] Fix another missing cast , makes the entire Mono test suite pass
[ profiler ] Add an assertion to the log profiler to catch the thread resurrection case .
2003 - 08 - 01 Andreas Nahr < ClassDevelopment @ A - SoftTech . com >
Make things work again .
2009 - 09 - 27 Gonzalo Paniagua Javier < gonzalo @ novell . com >
2010 - 06 - 15 Gonzalo Paniagua Javier < gonzalo @ novell . com >
2003 - 03 - 07 Alp Toker < alp @ atoker . com >
2009 - 03 - 03 Rodrigo Kumpera < rkumpera @ novell . com >
2009 - 02 - 27 Zoltan Varga < vargaz @ gmail . com >
* AssemblyInfo . cs : Mark [ CLSCompliant ( true ) ] , [ ComVisible ( false ) ] .
2003 - 07 - 31 Ben Maurer < bmaurer @ users . sourceforge . net >
Don ' t apply MONO_ATTR_FORMAT_PRINTF to mono_error_set_argument ( )
Fixes to nunit results xslt transform
Lock around critical data structures , patch from Jaroslaw Kowalski < jarek @ atm . com . pl >
2003 - 07 - 12 Atsushi Enomoto < ginga @ kit . hi - ho . ne . jp >
In . :
2008 - 09 - 22 Zoltan Varga < vargaz @ gmail . com >
* MdiChildContext . cs : Use the new borders instead of drawing them
2009 - 04 - 16 Rodrigo Kumpera < rkumpera @ novell . com >
Add test
Regex . IsMatch ( " abcab " , @ " ( ? < ab > ab ) c \ 1 " )
Use find_prev_seq_point ( ) for finding the IL offset for the top frame , mono_debug_il_offset_from_address ( ) doesn ' t seem to be precise enough . Fixes .
Merge pull request from akoeplinger / mkbundle - messages
Fix
* runtime / Makefile . am ( $ ( monoone_DATA ) $ ( monotwo_DATA ) $ ( monobins_DATA ) ) :
Wed Sep 5 15 : 54 : 47 CEST 2001 Paolo Molaro < lupus @ ximian . com >
Revert " Revert " Merge pull request from kumpera / wasm - target - cleanup " "
2006 - 03 - 13 Mike Kestner < mkestner @ novell . com >
add PageDelegate structure for passing delegates between appdomains
2008 - 04 - 16 Jb Evain < jbevain @ novell . com >
2008 - 07 - 10 George Giolfan < georgegiolfan @ yahoo . com >
Add System . Threading . Tasks . Dataflow assembly to the 4 . 0 build
Add the ' # if NET_2_0 ' conditional everywhere .
* CodeGenerator . cs : fixes for fall_true & fall_false for taking care of ast when is a Expre \ ssion .
update template wrt to 100712
2010 - 03 - 04 Atsushi Enomoto < atsushi @ ximian . com >
2007 - 09 - 26 Ivan N . Zlatev < contact @ i - nz . net >
Flush
* TypeTest . cs : Added test for Equals with null argument .
[ runtime ] Disable a test which asserts because of an abi break in recent xcode / clang versions . Works around .
2007 - 07 - 08 Zoltan Varga < vargaz @ gmail . com >
2008 - 09 - 23 Jb Evain < jbevain @ novell . com >
2003 - 05 - 16 Atsushi Enomoto < ginga @ kit . hi - ho . ne . jp >
Enable more thread apis
[ System ] : New X509Helper2 APIs for BTLS .
2008 - 09 - 07 Zoltan Varga < vargaz @ gmail . com >
In Parallel . For , returns immediately if the initial bound is superior or equal to the upper bound .
* mono / tests / appdomain - unload . cs ( Main ) : Pass along arguments to TestDriver .
[ corlib ] Preserve System . RuntimeType .
Flush
added VsaGlobalItem . cs
Fix .
* Makefile . am : Build ` docs ` after ` runtime ` , so that it can depend
[ wasm ] Implement GC support . Embedder must provide main loop pumping function request_gc_cycle .
2004 - 03 - 11 Martin Baulig < martin @ ximian . com >
Avoid crash when clicking a DataGrid row header with no data or columns . Fixes Xamarin bug 5487 .
[ HttpWebRequest ] Ignore Transfer - Encoding when there ' s no content .
2003 - 08 - 14 Atsushi Enomoto < ginga @ kit . hi - ho . ne . jp >
[ coop ] On OSX , set native_handle to the mac port .
Handle special case when disposing a socket .
Added refernce to Novell . Directory . Ldap
[ netcore ] Exclude some unused icalls ( )
Wed Feb 23 18 : 54 : 26 CET 2005 Paolo Molaro < lupus @ ximian . com >
2002 - 01 - 03 Nick Drochak < ndrochak @ gol . com >
2008 - 11 - 26 Atsushi Enomoto < atsushi @ ximian . com >
Ignore dll ' s and pdb ' s
[ arm ] Do not use the ARM Run - time ABI in cross - compilers ( )
Merge pull request from BrzVlad / fix - drainstack - stats
Fix the test setup , Run was using the wrong test
2005 - 07 - 17 Iain McCoy < iain @ mccoy . id . au >
2004 - 05 - 27 Gonzalo Paniagua Javier < gonzalo @ ximian . com >
Set invariant DateTimeFormatInfo for all invariant format specifiers . Fixes
Added more tests for statements
Bump bockbuild to bump gtksharp ( )
Check if number format info properties are empty first .
Removed the use of signals to control the profiler and extended the socket based interface to control also heap snapshots .
2010 - 01 - 19 Atsushi Enomoto < atsushi @ ximian . com >
2002 - 05 - 21 Martin Baulig < martin @ gnome . org >
fixed more srtncpy bugs
2008 - 04 - 24 Zoltan Varga < vargaz @ gmail . com >
Flush
2009 - 02 - 08 Gonzalo Paniagua Javier < gonzalo @ novell . com >
2006 - 11 - 01 Atsushi Enomoto < atsushi @ ximian . com >
[ w32handle ] Unify WaitHandle . Wait { One , Any , All } icalls ( )
2005 - 08 - 19 Ankit Jain < jankit @ novell . com >
[ jit ] Implement support for atomic intrinsics on arm .
* * * empty log message * * *
2002 - 08 - 29 Dietmar Maurer < dietmar @ ximian . com >
Microsoft . Build . Tasks : vbnc2 must be used for the 2 . 0 profile
Those files aren ' t needed any more ( they are replaced by nunit tests ) .
Revert " [ assembly ] Add Facades dir to default_path on all runtimes "
2010 - 05 - 13 Zoltan Varga < vargaz @ gmail . com >
2010 - 01 - 13 Atsushi Enomoto < atsushi @ ximian . com >
2010 - 03 - 04 Rodrigo Kumpera < rkumpera @ novell . com >
2009 - 05 - 26 Rodrigo Kumpera < rkumpera @ novell . com >
Merge pull request from JamesB7 / system - security4
2005 - 12 - 20 Alexander Olk < alex . olk @ googlemail . com >
Update locale tables - add Tatar CLDR files tt . xml and tt_RU . xml ( Fixes ) ( )
2007 - 12 - 06 Atsushi Enomoto < atsushi @ ximian . com >
Fix typos
2009 - 03 - 16 Atsushi Enomoto < atsushi @ ximian . com >
passing this one
Fixed member order in Dictionary key and item in XamlObjectReader .
[ runtime ] mono_get_method_constrained_with_method now uses MonoError .
[ runtime ] Setup stack trace for use in exception filters too ( )
Reset thread abort exception for internal threads regardless of unhandled exception policy .
2004 - 08 - 16 Zoltan Varga < vargaz @ freemail . hu >
Make NestedFuture test faster
Auto detect BOM - less utf - 16 and utf - 32 . Fixed bug .
added author comment
2007 - 12 - 27 Jonathan Pobst < monkey @ jpobst . com >
2009 - 04 - 19 Zoltan Varga < vargaz @ gmail . com >
[ msvc ] Update csproj files
2008 - 11 - 09 Zoltan Varga < vargaz @ gmail . com >
2007 - 06 - 13 Dick Porter < dick @ ximian . com >
2003 - 12 - 08 Ben Maurer < bmaurer @ users . sourceforge . net >
2008 - 09 - 28 Miguel de Icaza < miguel @ novell . com >
2008 - 07 - 04 Andreas Nahr < ClassDevelopment @ A - SoftTech . com >
2005 - 05 - 18 Zoltan Varga < vargaz @ freemail . hu >
[ System ] Fixed an error in the implementation of System . Net . IPAddress . IsIPv6Teredo
* * * empty log message * * *
Fix interpreter mini regression test on Windows x64 .
* X11Dnd . cs : Add support for handling the QueryContinue and
Avoid a crash in if image - > references [ i ] is REFERENCE_MISSING .
2010 - 07 - 14 Sebastien Pouliot < sebastien @ ximian . com >
merge 99762 : 100015
MonoTests . System . Xml : corrected expected / actual parameters position in Assert . AreEqual method .
2006 - 04 - 25 Atsushi Enomoto < atsushi @ ximian . com >
* ModuleBuilderTest . cs : Added test for DefineType with interface as
[ merp ] Remove unnecessary call to msync in mono_state_free_mem ( )
oops , removed commented code
2005 - 06 - 06 Zoltan Varga < vargaz @ freemail . hu >
* Image . jvm . cs : fixed redolution properties
2009 - 06 - 10 Marek Safar < marek . safar @ gmail . com >
[ metadata ] Move mono_class_compute_gc_descriptor back to object . c
2002 - 05 - 05 Tim Coleman < tim @ timcoleman . com >
Fix a warning .
Enable - - aot ( but not - - aot = full ) on osx / x86
Add Gtk # and Monodoc links for Mandrake 9 . 1
2005 - 01 - 11 Peter Bartok < pbartok @ novell . com >
ContainerControl . cs : Raise Validation and Enter / Leave events
2007 - 06 - 20 Marek Habersack < mhabersack @ novell . com >
* monodis . 1 : Document - - blob
[ sdks ] Disable the interpreter on ios device builds . ( )
Update corefx submodule for AIX readdir_r fix ( )
Update after merge conflicts
[ System ] Improved " Could not abort " exception .
2004 - 08 - 29 Nick Drochak < ndrochak @ ieee . org >
commits - to - changelog script .
2005 - 07 - 07 Atsushi Enomoto < atsushi @ ximian . com >
[ runtime ] Add missing semicolon in mono_rand_try_get_bytes ( )
* 5 - 6 - 2002 Dennis Hayes < dennish @ raytek . com >
Patches from Daniel Campos
2009 - 11 - 26 Marek Safar < marek . safar @ gmail . com >
svn path = / trunk / mcs / ; revision = 66124
2002 - 12 - 27 Sebastien Pouliot < spouliot @ videotron . ca >
2007 - 05 - 31 Carlos Alberto Cortez < calberto . cortez @ gmail . com >
2010 - 06 - 06 Carlos Alberto Cortez < calberto . cortez @ gmail . com >
[ aot ] Emit the same instances in hybrid aot mode as in full aot mode .
Reduce warning level for external libraries
2005 - 07 - 31 Atsushi Enomoto < atsushi @ ximian . com >
[ MonoError ] Add ArgumentNull error case
2009 - 10 - 21 Gonzalo Paniagua Javier < gonzalo @ novell . com >
* Mono . Security_test . args : Added
[ jit ] Avoid running class cctors during AOT when using the AggressiveInlining attribute . Fixes .
In mcs :
2009 - 02 - 10 Zoltan Varga < vargaz @ gmail . com >
Fix bug - - dupe and dup are quite different . ( )
[ netcore ] ifdef out unused icalls / files . ( )
Minor fixes .
2004 - 12 - 01 Atsushi Enomoto < atsushi @ ximian . com >
The CopyFrom methods in these classes shouldn ' t copy a value that is equal
* FileDialog . cs :
[ mkbundle ] Bundle unmanaged libs
Enable sgen support for s390x
[ MacSDK ] Update F # - > 4 . 1 . 8
added myself
Mirror changes from mono / corefx , corert , coreclr ( )
2002 - 04 - 16 Gaurav Vaish < Gaurav . Vaish @ iitk . ac . in >
System . Web . SessionState : simplify sessions retrival polling
Revert " [ jit ] Emit code to throw method access exceptions instead of throwing them from the JIT , so the stack trace is correct . Fixes . "
2006 - 08 - 17 Duncan Mak < duncan @ a - chinaman . com >
2001 - 12 - 30 Nick Drochak < ndrochak @ gol . com >
[ netcore ] Bump roslyn version
Extend supported XmlDateTimeSerialization formats . Fixed bug .
2008 - 02 - 03 Carlos Alberto Cortez < calberto . cortez @ gmail . com >
2002 - 05 - 14 Gonzalo Paniagua Javier < gonzalo @ ximian . com >
[ amd64 ] Unify the register saving / restoring code for the fp / no - fp cases .
2005 - 10 - 23 Sebastien Pouliot < sebastien @ ximian . com >
tests may not use the cwd for temp files
[ loader ] Fix off by 1 ( netstandard is a framework facade ) ( )
Merge pull request from esdrubal / httpfix
Reduced conditional expression requires both sides to be constant
2005 - 08 - 28 Zoltan Varga < vargaz @ gmail . com >
2008 - 08 - 28 Zoltan Varga < vargaz @ gmail . com >
2005 - 01 - 19 Martin Baulig < martin @ ximian . com >
Prepping for monodoc move from monodoc / engine to mcs / class / monodoc . . .
General Enum .
Use d14 and d15 as scratch VFP registers .
2004 - 03 - 17 Jesper Pedersen < jep @ itplus . dk >
[ interp ] fix warning introduced with 8e0a12189a84 ( )
2004 - 03 - 28 Zoltan Varga < vargaz @ freemail . hu >
Disable included libgc completely untils it ' s fixed .
Merge pull request from BogdanovKirill / webrequesttest
Fix the build .
* AssemblyNameTest . cs : Added tests for Clone and serialization without
New test .
* Removed AllTests . cs , System . Data / AllTests . cs ,
[ profiler ] Some improvements to the code coverage API .
In class / Microsoft . Build . Engine / Microsoft . Build . BuildEngine :
[ System . ServiceModel ] return the same object reference from MessageVersion . CreateVersion
another test
2002 - 09 - 11 Dick Porter < dick @ ximian . com >
2003 - 05 - 16 Dick Porter < dick @ ximian . com >
string format implementation
2005 - 02 - 15 Atsushi Enomoto < atsushi @ ximian . com >
2006 - 11 - 08 Sebastien Pouliot < sebastien @ ximian . com >
2003 - 10 - 10 Zoltan Varga < vargaz @ freemail . hu >
Build under different configurations
[ bcl ] Update BCL Linked Size ( )
Remove duplicate
I had some hack in this library too .
2004 - 10 - 04 John Luke < john . luke @ gmail . com >
2005 - 10 - 14 Senganal T < tsenganal @ novell . com >
fix type
Add
Adding System . IO . StreamReaderTest . Only non - inherited methods are tested .
2004 - 04 - 22 Gonzalo Paniagua Javier < gonzalo @ ximian . com >
New tests .
2009 - 08 - 24 Atsushi Enomoto < atsushi @ ximian . com >
new stub checkin .
remove debug code
2009 - 03 - 31 Carlos Alberto Cortez < calberto . cortez @ gmail . com >
Fix : Not replace an escaped left bracket inside of captured group .
* System . Web . vmwcsproj : grasshopper project file fix , added PageParser . jvm . cs , WebServiceParser . jvm . cs
2002 - 08 - 12 Gaurav Vaish < gvaish_mono @ lycos . com >
Add . gitignore file in mono / arch / arm64 .
[ interp ] Add assert in jmp that sp = = stack at start . ( )
2004 - 01 - 22 Atsushi Enomoto < atsushi @ ximian . com >
Fixed typo
2007 - 10 - 07 Jb Evain < jbevain @ novell . com >
Fix compiler warnings .
2004 - 05 - 11 Dick Porter < dick @ ximian . com >
[ sgen ] Don ' t pump the hazard free queue from sgen , the finalizer thread does it now .
Include build_llvm_config . sh in tarballs ( Closes https : / / github . com / mono / mono / issues / 9456 )
2003 - 06 - 19 Nick Drochak < ndrochak @ gol . com >
2006 - 04 - 12 Sebastien Pouliot < sebastien @ ximian . com >
2006 - 05 - 15 Peter Dennis Bartok < pbartok @ novell . com >
Minor fixes
refactoring :
Add Marek to the about section
2008 - 03 - 06 Jb Evain < jbevain @ novell . com >
2002 - 08 - 18 Dick Porter < dick @ ximian . com >
Fix __asm__ specifications for Clang ( )
2004 - 09 - 03 Sebastien Pouliot < sebastien @ ximian . com >
Enabled - - preload - files without AOT ( )
Fix README . md to show correct values for - - with - libgc
DrawingTest . cs : Added special tolerance feature
[ winforms ] Cope with null Hwnd value from Hwnd . ObjectFromHandle
Fix a few spelling mistakes
* ILParser . jay : New grammar that I haven ' t made a mess of , this
2006 - 05 - 22 Marek Sieradzki < marek . sieradzki @ gmail . com >
[ jit ] Align float / double Nan / Inf to uint conversion behavior with CoreCLR / LLVM / Roslyn ( )
2005 - 09 - 30 Zoltan Varga < vargaz @ gmail . com >
move SocketException_2_1 to System . Net hopefully fix the build
Added regression tests as visual studio projects .
2004 - 07 - 08 Sebastien Pouliot < sebastien @ ximian . com >
Added attributes .
[ SDKS ] fix Linux ( and possibly other OSes ) build for Android ( )
Only return true on Windows for Uri . IsUnc on file : / / URI
[ aot ] Initialize the temporary path to an empty string .
2006 - 03 - 27 Robert Jordan < robertj @ gmx . net >
Tue Sep 24 11 : 27 : 38 CEST 2002 Paolo Molaro < lupus @ ximian . com >
Merge pull request from saper / shellredir
2003 - 04 - 02 Miguel de Icaza < miguel @ ximian . com >
2009 - 04 - 06 Zoltan Varga < vargaz @ gmail . com >
2009 - 07 - 02 Jb Evain < jbevain @ novell . com >
[ gc ] Introduce mono_object_new_mature in the same fashion as the other alloc functions .
[ runtime ] Remove dependency on getdtablesize
2005 - 01 - 14 Zoltan Varga < vargaz @ freemail . hu >
Fix another stack alignment issue under osx - x86 .
2005 - 07 - 13 Atsushi Enomoto < atsushi @ ximian . com >
2007 - 02 - 21 Nagappan A < anagappan @ novell . com >
2009 - 10 - 04 Mark Probst < mark . probst @ gmail . com >
[ runtime ] Fix monitor exception throwing
SL2 [ SecurityCritical ] data
Merge pull request from directhex / master
* RichTextBox . cs : Set the line endings correctly , when flushing
2008 - 10 - 17 Marek Safar < marek . safar @ gmail . com >
2008 - 01 - 16 Jb Evain < jbevain @ novell . com >
2005 - 09 - 27 Martin Baulig < martin @ ximian . com >
DeflateStream : check IsInvalid before disposing ( )
[ Mono . Security ] Clear Certificates after Import or Remove [ part of PR ]
* PagedDataSource . cs ( PageCount ) : Return page count of 1 if there
[ profiler ] Bump log profiler version to 2 . 0 due to breaking argument changes .
* DbConnectionStringBuilder . cs ( ICollection . CopyTo ) : Use
2009 - 02 - 22 Gonzalo Paniagua Javier < gonzalo @ novell . com >
2004 - 11 - 23 Geoff Norton < gnorton @ customerdna . com >
Remove an unneccesary assert which is hit when a double is passed in split reg - stack in soft - float mode on non - EABI arm .
Make * mcs compiler aliases differ in sdk version only
Merge pull request from lateralusX / lateralusX / windows - x64 - llvm - support - master
2004 - 08 - 23 Martin Baulig < martin @ ximian . com >
* HttpRequest . cs ( CheckString ) : add back in the check for \ xff1c
Merge pull request from alexanderkyte / conflicting_attrs
2010 - 05 - 03 Carlos Alberto Cortez < calberto . cortez @ gmail . com >
2005 - 12 - 08 Sebastien Pouliot < sebastien @ ximian . com >
Add tests for SerializationBinder . BindToName usage with BinaryFormatter .
2006 - 03 - 14 Martin Baulig < martin @ ximian . com >
2002 - 12 - 17 Ville Palo < vi64pa @ koti . soon . fi >
2006 - 11 - 27 Jonathan Chambers < joncham @ gmail . com >
2010 - 05 - 21 Atsushi Enomoto < atsushi @ ximian . com >
2008 - 10 - 16 Rodrigo Kumpera < rkumpera @ novell . com >
LinePragmas default to true if value is not present
2007 - 10 - 27 Zoltan Varga < vargaz @ gmail . com >
2008 - 01 - 27 Igor Zelmanovich < igorz @ mainsoft . com >
2007 - 05 - 30 Sebastien Pouliot < sebastien @ ximian . com >
2006 - 05 - 23 Atsushi Enomoto < atsushi @ ximian . com >
2010 - 04 - 12 Carlos Alberto Cortez < calberto . cortez @ gmail . com >
2010 - 07 - 27 Atsushi Enomoto < atsushi @ ximian . com >
2005 - 09 - 16 Kazuki Oikawa < kazuki @ panicode . com >
oops
Get rid of the ' is_pinvoke ' argument to get_call_info ( ) , use sig - > pinvoke instead .
2006 - 08 - 25 Sebastien Pouliot < sebastien @ ximian . com >
2008 - 04 - 21 Jonathan Pobst < monkey @ jpobst . com >
2005 - 09 - 28 Marek Safar < marek . safar @ seznam . cz >
update . csproj .
[ sgen ] Remove unused parameter
[ system ] Add DeliveryFormat to SmtpSection . Fixes
Add 4 . 0 FormatterServices . IGetSurrogateForCyclicalReference .
2009 - 07 - 22 Atsushi Enomoto < atsushi @ ximian . com >
Update test
Fri Aug 2 11 : 40 : 32 CEST 2002 Paolo Molaro < lupus @ ximian . com >
* xbuild / Microsoft . Common . targets ( BuildingResources ) : Refactor to
2003 - 05 - 05 Gonzalo Paniagua Javier < gonzalo @ ximian . com >
2008 - 07 - 16 Zoltan Varga < vargaz @ gmail . com >
* ContainerControl . cs : Set active_control even earlier , before
[ sgen ] Tag cemented objects so we don ' t have to go through the hash .
okay , add the System . Web . Profile stuff back to the build ( hopefully it won ' t break it this time ) .
[ mini ] Memory barrier fixes , simplifications , and optimizations .
New test .
* generic . cs ( TypeParameter ) : Implement IMemberContainer .
2002 - 05 - 17 Dietmar Maurer < dietmar @ ximian . com >
Bump mono to 3 . 0 . 3
[ corlib ] Removed TimeZone unused methods .
Fix the loaded llvm build .
2002 - 10 - 31 Ville Palo < vi64pa @ koti . soon . fi >
[ mono - threads ] Factor common platform create thread code ( )
Merge pull request from cdrnet / fix - biginteger - gcd
[ monkeydoc ] Correctly process arguments given to the ecma html - izer
Fri Jan 17 21 : 14 : 18 CET 2003 Paolo Molaro < lupus @ ximian . com >
2009 - 12 - 13 Miguel de Icaza < miguel @ novell . com >
[ corlib ] Fix flaky ThreadPoolTests . AsyncLocalCapture test ( )
* Page . jvm . cs : optimized SetupResponseWriter , fixed getChildCount and getChildren methods
* HttpResponse . cs : fixed TARGET_J2EE part of Redirect for portlet
2008 - 07 - 06 Sebastien Pouliot < sebastien @ ximian . com >
2009 - 10 - 24 Zoltan Varga < vargaz @ gmail . com >
[ mini ] fix memory dumping on devices ( )
[ corlib ] Runtime attributes from reference sources
2010 - 06 - 14 Atsushi Enomoto < atsushi @ ximian . com >
2008 - 05 - 30 Marek Safar < marek . safar @ gmail . com >
2006 - 07 - 21 Martin Baulig < martin @ ximian . com >
Make pdb2mdb work with files in other directories .
* SingleConverter . cs : Implement conversion from string to match MS . NET .
* * * * Merged r41599 from MCS * * * *
[ WinForms ] Editing the DialogResult of the form in the button click event was ignored . ( )
[ interp ] Fix Assertion at threads . c : 326 , condition ` new_val > = 0 ' not met ( )
inline variable
New test .
2005 - 01 - 19 Sebastien Pouliot < sebastien @ ximian . com >
add zlib functions
2008 - 08 - 05 Jb Evain < jbevain @ novell . com >
Make the AOT compiler stats output a bit more compact .
[ tests ] gc - descriptors : Only copy file in case of VPATH build
2004 - 10 - 27 Chris Toshok < toshok @ ximian . com >
Fix SemaphoreSlim Constructor .
[ build ] Fix the build for NET_4_5 ( and maybe other profiles ) from the pull request 981
Another build fix
2010 - 06 - 06 Zoltan Varga < vargaz @ gmail . com >
2010 - 01 - 22 Marek Habersack < mhabersack @ novell . com >
Appdomain unloading for SGen .
Implement some async System . Net . Http methods
Add None to RegistryValueKind enumeration .
2003 - 02 - 08 Sebastien Pouliot < spouliot @ videotron . ca >
2008 - 08 - 13 Marek Habersack < mhabersack @ novell . com >
2007 - 03 - 07 Zoltan Varga < vargaz @ gmail . com >
Update
2004 - 01 - 05 Zoltan Varga < vargaz @ freemail . hu >
Oopsety
VS signature update
Store browscap . ini as a text file
[ runtime ] Switch GENERATE_GET_CLASS_WITH_CACHE to use mono_class_load_from_name . Kill dead macro .
Fix broken finally blocks in the binaryreader tests
oops .
2005 - 11 - 03 Gonzalo Paniagua Javier < gonzalo @ ximian . com >
Add missing prototypes to supportw . c .
[ bcl ] Update BCL Linked Size
Add space
Deleting all the files starting with lower case ' a ' .
2002 - 07 - 22 Tim Coleman < tim @ timcoleman . com >
Merge pull request from kumpera / fix_ji_free
Skip load if same register
Revert " Merge pull request from methane / websocket "
* ILParser . jay : When looking up types first check to see if this
2004 - 11 - 08 Atsushi Enomoto < atsushi @ ximian . com >
[ System . Data ] referencesource - > corefx ( )
[ mono - api - html ] Output C # ' char ' instead of ' System . Char '
2009 - 11 - 04 Marek Safar < marek . safar @ gmail . com >
2007 - 11 - 09 Jb Evain < jbevain @ novell . com >
2002 - 12 - 15 Sebastien Pouliot < spouliot @ videotron . ca >
2004 - 05 - 21 Atsushi Enomoto < atsushi @ ximian . com >
2003 - 01 - 30 Ville Palo < vi64pa @ koti . soon . fi >
[ mini ] Add missing include to mini - exceptions - native - unwinder . c
2009 - 04 - 28 Zoltan Varga < vargaz @ gmail . com >
2004 - 02 - 05 Nick Drochak < ndrochak @ gol . com >
2004 - 03 - 24 Sebastien Pouliot < sebastien @ ximian . com >
Fix an unlikely case in cmd_to_string ( ) where cmds would be accessed uninitialized .
[ corlib ] Make Task . Run specify TaskScheduler . Default and fix Task . Run ( Task < Task > )
2006 - 01 - 26 Atsushi Enomoto < atsushi @ ximian . com >
Comitted the wrong file before . Here ' s the correct one .
[ runtime ] Fix msvc build .
2005 - 11 - 23 Mike Kestner < mkestner @ novell . com >
* Makefile ( all - profiles $ ( STD_TARGETS : = - profiles ) ) : Make
Fix a warning .
2010 - 03 - 05 Rodrigo Kumpera < rkumpera @ novell . com >
[ acceptance - tests ] Don ' t mark msbiology as a failure if it OOMs on 32 - bit .
[ sgen ] Never mark from two threads at the same time .
In . :
In class / Microsoft . Build . Utilities / Microsoft . Build . Utilities :
add TODO
[ System ] : Make HttpListener partial and move Mono - specific bits into HttpListener . Mono . cs . ( )
2005 - 04 - 18 Sebastien Pouliot < sebastien @ ximian . com >
* Menu . cs : Make sure the parent of the items get set correctly
* OdbcType . cs : fixed linefeeds to DOS to match other sources
Refactor Add / Take / TryAdd / TryTake methods into a common block in BlockingCollection
[ runtime ] Replace pedump boehm dependency with sgen dependency
2002 - 10 - 01 Rodrigo Moya < rodrigo @ ximian . com >
2005 - 09 - 05 Martin Baulig < martin @ ximian . com >
2009 - 01 - 16 Miguel de Icaza < miguel @ novell . com >
[ xbuild ] Add workaround for pre - 2 . 0 csc and convert full to portable debug mode
In ilasm / codegen :
2005 - 11 - 17 Zoltan Varga < vargaz @ gmail . com >
2005 - 12 - 17 Alexander Olk < alex . olk @ googlemail . com >
[ utils ] Fix a bad format specifier .
New build system from Peter Williams ( peter @ newton . cx )
* man / gacutil . 1 : Basic man page for gacutil . This doesn ' t include
2006 - 09 - 08 Gonzalo Paniagua Javier < gonzalo @ ximian . com >
2004 - 08 - 17 Ben Maurer < bmaurer @ users . sourceforge . net >
New tests
Remove some unused arguments from mono_arch_handle_exception ( ) / mono_handle_exception ( ) .
2004 - 10 - 10 David Sheldon < dave - mono @ earth . li >
* Stack . cs : Stupid bug fixes .
* AppDomainLevelActivator . cs : little fix .
2004 - 11 - 07 Zoltan Varga < vargaz @ freemail . hu >
2005 - 07 - 28 Lluis Sanchez Gual < lluis @ novell . com >
2003 - 12 - 13 Martin Baulig < martin @ ximian . com >
* PropertyGridView . cs : Use a message filter to check when to
Update release notes URLs in the NEWS file .
[ runtime ] MonoError - ize mono_array_new_cached
Revert " update rx to the latest rx - oss - v1 . 1 build . "
2010 - 05 - 26 Marek Habersack < mhabersack @ novell . com >
2010 - 07 - 05 Atsushi Enomoto < atsushi @ ximian . com >
2008 - 06 - 04 Zoltan Varga < vargaz @ gmail . com >
2007 - 04 - 27 Jonathan Pobst < monkey @ jpobst . com >
* XmlTextWriter . cs : when adding an attribute with a namespace , a prefix must be automaticaly
* Mono . Mozilla / WebBrowser . cs : create the browser window when the control gets
Sat Feb 15 15 : 15 : 03 CET 2003 Paolo Molaro < lupus @ ximian . com >
2005 - 07 - 07 Florian Gross < flgr @ ccan . de >
fixed member accessibility to match MS ( corcompare ) .
[ mcs ] Don ' t consider loads from captured local cheap
2007 - 05 - 15 Marek Habersack < mhabersack @ novell . com >
Add . gitignore
2005 - 11 - 04 Sebastien Pouliot < sebastien @ ximian . com >
2008 - 05 - 06 Jb Evain < jbevain @ novell . com >
2002 - 04 - 04 Dietmar Maurer < dietmar @ ximian . com >
[ interp ] Outline STRMFLD_VT . ( )
Another fix
Modified SQLDescribeCol to work around bug in pinvoke . . . when doing a ref on a typed enum
Fix XmlReaderSettings : : Clone
[ sgen ] Handle copy object failure .
* RealProxy . cs : Added some missing methods .
fix the fix
2004 - 08 - 17 Sebastien Pouliot < sebastien @ ximian . com >
Major name lookup fixes
New test .
[ runtime ] Fix the ARM build .
2002 - 07 - 08 Gonzalo Paniagua Javier < gonzalo @ ximian . com >
Fix gshared support in sdb .
2006 - 12 - 19 Marek Sieradzki < marek . sieradzki @ gmail . com >
Typo
2010 - 06 - 01 Zoltan Varga < vargaz @ gmail . com >
2006 - 12 - 19 Sebastien Pouliot < sebastien @ ximian . com >
Check in up - to - date version of RegexTest . cs . Somehow the previous
[ interpreter ] fix ves_array_set
2006 - 08 - 08 Sebastien Pouliot < sebastien @ ximian . com >
2005 - 04 - 11 James Willcox < james @ ximian . com >
Remove ArrayList dependency
P / invoke the win32 DLL name so Mono . Cairo will work on . NET
2003 - 08 - 06 Andreas Nahr < ClassDevelopment @ A - SoftTech . com >
* Control . cs : fixed AppRelativeTemplateSourceDirectory that should depend on AppRelativeVirtualPath
Re - enable a test for the MOBILE profile ( by ensuring the required method is not removed by the linker )
[ corlib ] Use String from CoreCLR ( )
2005 - 08 - 16 Martin Baulig < martin @ ximian . com >
2005 - 05 - 26 Gonzalo Paniagua Javier < gonzalo @ ximian . com >
simplifies object destruction and fixes
[ runtime ] Add back the locking to setup_interface_offsets ( ) to work around crashes in roslyn .
2008 - 09 - 12 Jb Evain < jbevain @ novell . com >
2008 - 06 - 24 Marek Safar < marek . safar @ gmail . com >
2004 - 10 - 28 Atsushi Enomoto < atsushi @ ximian . com >
[ threads ] avoid race between setting the state flag and suspension
2005 - 09 - 29 Martin Baulig < martin @ ximian . com >
2009 - 02 - 06 Gonzalo Paniagua Javier < gonzalo @ novell . com >
2009 - 03 - 18 Atsushi Enomoto < atsushi @ ximian . com >
* get . c ( get_encoded_user_string_or_bytearray ) : Add len param .
[ io - layer ] Cope with pids > 16 bits
Use context class loader , if available
Revert " [ bcl ] Fix System . Net . HttpWebRequestTest . GetRequestStream hang "
2009 - 01 - 14 Atsushi Enomoto < atsushi @ ximian . com >
2004 - 05 - 26 Atsushi Enomoto < atsushi @ ximian . com >
2004 - 01 - 12 Patrik Torstensson
2004 - 09 - 25 Zoltan Varga < vargaz @ freemail . hu >
update . csproj .
Fixety fix
* Makefile ( net_2_0_bootstrap / mcs . exe . config ) : Copy gmcs . exe . config .
* * * empty log message * * *
[ ci ] Add CI_TAGS handling for ccache ( )
* Mono . Documentation / XmlDocUtils . cs : GetInterfaces ( ) can throw if the
2006 - 08 - 15 Peter Dennis Bartok < pbartok @ novell . com >
Fix an NRE in System . Timers . Timer . Dispose ( ) .
2009 - 04 - 17 Rodrigo Kumpera < rkumpera @ novell . com >
If there is a crash and the debugger is attached , we call mini_get_dbg_callbacks ( ) - > send_crash ( ) . ( )
2004 - 12 - 01 Marek Safar < marek . safar @ seznam . cz >
[ runtime ] MonoError - ize mono_async_result_new
2010 - 02 - 18 Marek Habersack < mhabersack @ novell . com >
Enabled variance tests .
2005 - 01 - 30 Zoltan Varga < vargaz @ freemail . hu >
2009 - 05 - 21 Miguel de Icaza < miguel @ novell . com >
2005 - 08 - 26 Atsushi Enomoto < atsushi @ ximian . com >
* OdbcTransaction . cs : fixed signature to match MS . NET
Merge pull request from svick / patch - 1
2005 - 01 - 25 Martin Baulig < martin @ ximian . com >
2004 - 01 - 27 Sebastien Pouliot < spouliot @ videotron . ca >
Handle a loader generic type hierarchy corner case . Fixes bxc
PropertyInfoTest . ConstantValue ( ) shouldn ' t fail if temp file cannot be deleted
Reject varianly implemented interface members .
2010 - 06 - 05 Mark Probst < mark . probst @ gmail . com >
2007 - 10 - 17 Sebastien Pouliot < sebastien @ ximian . com >
[ xbuild ] Fix 635767 . Support enums for task property Outputs .
New test .
2005 - 07 - 30 Zoltan Varga < vargaz @ freemail . hu >
2002 - 11 - 28 DennisHayes < dennish @ raytek . com >
Added System . IO . Compression implementation .
2008 - 08 - 04 Jb Evain < jbevain @ novell . com >
Revert 12fdfa365c2aa449c132bf0e04f59393792479f0 and save / restore task status to also fix .
2004 - 10 - 13 Atsushi Enomoto < atsushi @ ximian . com >
[ io - layer ] Move WapiHandleType and WapiHandleOps to handles - private . h
2008 - 10 - 01 Marek Safar < marek . safar @ gmail . com >
2004 - 09 - 07 Ben Maurer < bmaurer @ users . sourceforge . net >
In Test / System . Xml :
Fix sdb trampoline on Windows x64
2009 - 02 - 05 Zoltan Varga < vargaz @ gmail . com >
2008 - 03 - 06 Jb Evain < jbevain @ novell . com >
Fix synchronized_generic_cache .
Bump expected LLVM version in configure .
2005 - 01 - 19 Miguel de Icaza < miguel @ ximian . com >
Implement Array . UnsafeLoad and Array . UnsafeStore . Both functions ignores bounds check when possible .
[ interp ] initial infrastructure for whitebox testing of optimization passes ( )
Fixed path starting with ' / ' on custom schemes , mailto and news .
IO threads have their own name
New tests for Regex :
Fix bug .
2004 - 02 - 03 Zoltan Varga < vargaz @ freemail . hu >
[ build ] Updated the class library MSBuild projects .
This is a patch to fix bug 34123 . System . Drawing . Printing . PrintingServicesUnix is failing to print on Mac OSX Yosemite because the CUPS function cupsTempFile now always returns an empty string . This function was deprecated . Now replacing cupsTempFile call with call to cupsTempFd to fix printing . I do not see a clear way to unit test this , and there appear to be no prior unit tests covering this . Essentially we are swapping a function call for another that returns the same type of result . This was tested on my copy of net_4_x master . The same fix could be applied to mono version 3 also . This change is released under the MIT license .
[ runtime ] Clean up an # ifdef block .
2005 - 01 - 19 Sureshkumar T < tsureshkumar @ novell . com >
[ profiler ] Fix several race conditions in accessing MonoProfilerThread . buffer .
[ cmake ] Add - DCHECKED_BUILD = args support , and check getters in Debug ( )
2010 - 05 - 16 Marek Habersack < mhabersack @ novell . com >
2007 - 08 - 04 Jb Evain < jbevain @ novell . com >
2007 - 09 - 06 Jonathan Pobst < monkey @ jpobst . com >
2010 - 03 - 19 Sebastien Pouliot < sebastien @ ximian . com >
[ interp ] Cleanup : Replace retval - > with result . and retval with & result . ( )
2003 - 12 - 22 Zoltan Varga < vargaz @ freemail . hu >
2008 - 06 - 17 Rodrigo Kumpera < rkumpera @ novell . com >
We can not obsolete in the 1 . 2 series
[ runtime ] Avoid loading marshal info for types without field marshallers when calling FieldInfo : get_marshal_info ( ) .
2009 - 06 - 11 Zoltan Varga < vargaz @ gmail . com >
Support call chains ( backtrace ) in the stat profiler .
In Test / System . Windows . Forms . Layout :
Clean up formatting
2009 - 01 - 21 Mark Probst < mark . probst @ gmail . com >
2009 - 06 - 22 Rodrigo Kumpera < rkumpera @ novell . com >
2009 - 07 - 14 Zoltan Varga < vargaz @ gmail . com >
Applied patch from Marek Safar to add more error checking .
[ runtime ] Avoid including mcs / build / config . make into mini / Makefile . am , it can overwrite all kinds of make variables . Pass DEFAULT_PROFILE normally as an autoconf variable .
[ mcs ] struct initializers are still experimental
2008 - 12 - 11 Marek Safar < marek . safar @ gmail . com >
[ WIP ] Make debugger agent startup faster ( )
Little fixes .
2006 - 12 - 18 Rolf Bjarne Kvinge < RKvinge @ novell . com >
time to set eol - style : native .
2005 - 08 - 28 Gonzalo Paniagua Javier < gonzalo @ ximian . com >
list . unix modified
Kill the MOONLIGHT define in Mono . Security .
[ corlib ] Improve the exception message in Delegate : : Remove .
INSTALL was renamed to INSTALL . txt so make install will on Windows
[ corlib ] Claims from reference sources
[ mcs ] Don ' t report any not applicable operator errors during constant folding .
Add regression test for
SendKeys X11 support .
2009 - 06 - 19 Rodrigo Kumpera < rkumpera @ novell . com >
[ sgen ] Use size_t consistently in the major collector for object sizes and number of memory blocks .
[ System ] : AndroidPlatform . CertStoreLookup ( ) : add to the build , but return null .
Updated with latest changes
* GridView : fixed DataKeyNames property , moved to ControlState
2004 - 11 - 18 Lluis Sanchez Gual < lluis @ novell . com >
Update man page
2009 - 01 - 10 Zoltan Varga < vargaz @ gmail . com >
Prepping for monodoc move from monodoc / engine to mcs / class / monodoc . . .
2004 - 11 - 05 Zoltan Varga < vargaz @ freemail . hu >
2005 - 09 - 27 Gonzalo Paniagua Javier < gonzalo @ ximian . com >
Move the find_pinning_reference ( ) function to sgen - debug . c .
[ profiler } Don ' t crash if a zero sampling frequency is given .
2007 - 10 - 26 Atsushi Enomoto < atsushi @ ximian . com >
2008 - 10 - 04 Gonzalo Paniagua Javier < gonzalo @ novell . com >
update
Funcs . cs : Fixed typo in Func ` 15 .
Remove Thread . [ Abort | Suspend | Resume ] from TvOS / WatchOS .
* Array . cs ( CopyTo ) : use GetLength ( ) instead of GetUpperBound ( )
Typo
2007 - 04 - 17 Atsushi Enomoto < atsushi @ ximian . com >
[ corlib ] ExceptionServices from reference sources
* XmlSerializationWriterTests . cs : Use Assert instead of deprecated
2001 - 08 - 29 Dietmar Maurer < dietmar @ ximian . com >
2007 - 10 - 19 Dick Porter < dick @ ximian . com >
[ llvm ] Fix atomic opcodes on ARM64 to emit correct fences .
Rework self suspend to avoid a race on Mach .
Fix a buffer overflow in the LLVM backend . Fixes .
Updated .
New test .
New tests
forgot this one .
* README : Update with details about the new build setup .
* Makefile ( DIST_ONLY_SUBDIRS ) : Add ' sqlsharp ' .
[ msbuild ] Bump to pick up test fixes
2005 - 10 - 17 Sebastien Pouliot < sebastien @ ximian . com >
First drop of a new Mono . Parallel assembly
2009 - 03 - 26 Jonathan Chambers < joncham @ gmail . com >
2004 - 08 - 26 Atsushi Enomoto < atsushi @ ximian . com >
Fix warnings .
[ runtime ] Move mono_aot_can_dedup into aot - runtime
* TemplateParser . cs : Ignore aspCompat attribute . This fixes bug
[ sgen ] Move time stuff to client code .
shhh
2003 - 03 - 31 Alan Tam < Tam @ SiuLung . com >
2008 - 05 - 20 Andreas Nahr < ClassDevelopment @ A - SoftTech . com >
* * * * Merged from MCS * * * *
2005 - 08 - 18 Gonzalo Paniagua Javier < gonzalo @ ximian . com >
Kill all executable flags on * . cs and ChangeLog files .
* * * * Merged r42573 from MCS * * * *
[ interp ] Put finally data back in locals instead of frame to save 16 bytes of stack . ( )
Update emscripten to 1 . 39 . 3 ( )
Clean up some whitespace in the output
Timer Tick events are now handed through Controls Async mechanism so the callbacks are executed in the same thread as X
Install mono - dl . h header
Fix IndexOutOfBounds exception , when removing and adding an item to another list
Fix IDE1005 ( simplify delegate invocation )
Moves StackTrace to shared partition ( dotnet / coreclr )
Revert " [ Mono . Security ] : Improve error reporting when sending TLS Alerts fail . "
2007 - 09 - 18 Marek Habersack < mhabersack @ novell . com >
2004 - 01 - 20 Atsushi Enomoto < atsushi @ ximian . com >
2001 - 11 - 22 Ravi Pratap < ravi @ ximian . com >
[ docs ] Update formatting in mono - api - type .
[ mobile_static ] Use runtime_preset configuration option
Just a stub to run some apps
2010 - 02 - 05 Rodrigo Kumpera < rkumpera @ novell . com >
2003 - 03 - 06 Sebastien Pouliot < spouliot @ videotron . ca >
2002 - 05 - 24 Dietmar Maurer < dietmar @ ximian . com >
Merge pull request from kumpera / com - interop - fix
2005 - 01 - 16 Zoltan Varga < vargaz @ freemail . hu >
2008 - 03 - 10 Ivan N . Zlatev < contact @ i - nz . net >
NUnitLite does not know Assert . IsInstanceOfType ( ) .
publish the same events as pthread_stop_world
2009 - 11 - 26 Marek Safar < marek . safar @ gmail . com >
2007 - 09 - 28 Atsushi Enomoto < atsushi @ ximian . com >
2006 - 02 - 27 Marek Safar < marek . safar @ seznam . cz >
2003 - 05 - 21 Zoltan Varga < vargaz @ freemail . hu >
Add feature define for multiple appdomains .
2008 - 10 - 31 Gonzalo Paniagua Javier < gonzalo @ novell . com >
[ Coop ] A bunch of nohandles ( ) . ( )
Use an enum instead of magic numbers to make the code more readable .
2003 - 09 - 30 Zoltan Varga < vargaz @ freemail . hu >
Fix more buglets exposed by the MOMA tool .
[ interp ] bail out on __arglist and propagate MonoError properly in transformer
2004 - 10 - 14 Atsushi Enomoto < atsushi @ ximian . com >
* VBCodeProviderTest . cs : Fixed line endings
2003 - 04 - 14 Miguel de Icaza < miguel @ ximian . com >
[ System ] Fix handle leak in Process : : GetProcess .
[ jit ] Remove unused defines .
Add some documentation to BlockingCollection
2009 - 10 - 13 Bill Holmes < billholmes54 @ gmail . com >
Run all the WebTests in one and only one TreadPool thread
2009 - 06 - 08 Zoltan Varga < vargaz @ gmail . com >
[ threads ] Remove mono_thread_notify_pending_exc_fn dead code
Commented out debugging message .
2005 - 02 - 25 Atsushi Enomoto < atsushi @ ximian . com >
2006 - 12 - 12 Atsushi Enomoto < atsushi @ ximian . com >
2005 - 04 - 29 Martin Baulig < martin @ ximian . com >
[ coop ] Add support for entering / exiting GC safe / unsafe region for products
2009 - 06 - 12 Atsushi Enomoto < atsushi @ ximian . com >
* MonoWSDL . cs : Implemented support for WSDL documents that import other
Add some new async methods
Fix typo .
2008 - 04 - 11 Atsushi Enomoto < atsushi @ ximian . com >
Sequence of comparison expressions ( A = = B = = C ) is invalid . Avoid ANE on property evaluation .
2005 - 09 - 02 Cesar Lopez Nataren < cnataren @ novell . com >
Merge pull request from akoeplinger / unify - test - namespaces
2006 - 05 - 16 Chris Toshok < toshok @ ximian . com >
* System . Configuration . dll . sources , System . Configuration20 . vmwcsproj : added OrdinalComparer . cs
[ profiler ] Make ` perf ` syscall work on ARM64 .
Handle OOM on string intern icall .
2005 - 07 - 23 Atsushi Enomoto < atsushi @ ximian . com >
* ProtocolReflector . cs : throw if not conformant , but declared to be .
Fix error number
[ wasm ] [ debugger ] Fixing step over in an async method ( )
* ContextsTest . cs , CallSeq . cs , ContextHookAttribute . cs , ServerObject . cs : Added .
Fix encode / decode_sleb128 ( ) on 64 bit systems .
2008 - 10 - 17 Zoltan Varga < vargaz @ gmail . com >
fix build
2006 - 12 - 31 Miguel de Icaza < miguel @ novell . com >
2008 - 05 - 05 Andreas Faerber < andreas . faerber @ web . de >
2004 - 03 - 17 Martin Baulig < martin @ ximian . com >
Update to the latest IKVM reflection
merge - r 60439 : 60440
[ Mono . Posix ] Add file that was on the xamarin - android repo so it can be built from mono .
2010 - 04 - 02 Zoltan Varga < vargaz @ gmail . com >
[ man ] Updated Mono man page with AOT gen - seq - points - file option .
Improve libgdiplus path handling
* ToolTask . cs ( ExecuteTool ) : Use LogToolCommand , instead of
oops , forgot a ;
[ x86 ] Fix the epilog tls access on linux too .
Bump API snapshot submodule
* ListView . cs : Add fireEvent argument to ReorderColumn since the
2005 - 01 - 27 Zoltan Varga < vargaz @ freemail . hu >
2008 - 06 - 13 Mark Probst < mark . probst @ gmail . com >
[ corlib ] Disable broken tests that depend on crazy precise scheduling and system timers .
2008 - 08 - 20 Zoltan Varga < vargaz @ gmail . com >
2006 - 10 - 06 Gonzalo Paniagua Javier < gonzalo @ ximian . com >
[ interp ] Defer local offset allocation to compacted emit phase ( )
2005 - 06 - 01 Gonzalo Paniagua Javier < gonzalo @ ximian . com >
Apply this here as well
