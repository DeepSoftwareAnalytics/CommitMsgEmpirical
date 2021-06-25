merge pull request from taylorwood master
not equals test
don t assign _largebytebuffer until clear it will be used
add test for duplicatepropertynamehandling replace
fix du converter exception typo
add backwards compatible enumutils trytostring call
fix unit test in according to the name
add deserialization test and cleaned up the tests
fix bad using in last commit
optimise linqbridge count
bug unexpected token comment
update jsonconvert cs
update javascriptutils cs
fix metro cert verification error
merge pull request from sgnr master
replaced containskey and getting by value with trygetvalue method
update jsontextreader cs
simplify lambda to expression
fix wrong define is used in stringutils tolower
add xml content encoding test
crlfs and newline
merge pull request from recycleddan master
fix hopefully metadatatypeattribute not being found on some platforms
fix loading extra assemblies by only filtering for byref types on default members
brackets have no case
fix medium trust error
pass cancellationtoken in jobject and jconstructor writetoasync
changed jsoncontracttype to always have all options
fix documentation for jvalue createundefined
merge pull request from christianrondeau camel casing with numbers
for respect unset initial value on jsonobjectattribute
fix using simple type names on silverlight
fix invalid cast with datetimeoffset
adds test for deserializing xmlelement and a minor change in the exception message
fix metro build
fix deserializexnode in net core
not correctly resizing internal string buffer when parsing long json documents
fix doc typo
optimise decimal predicate sum
add securitytransparent attribute to metro build
adds support of xmlelement as one of the allowed objecttype
fix elusive intellitrace runtime destabilization error
fix metro build
fix build error
optimise linqbridge select
optimise linqbridge any
optimise linqbridge enumerable where
fix dynamic serializing in silverlight
fix jtoken line info set
missing nullability annotation
fix typo in documentation
use cancellationtoken in two places it was missed
remove allowpartiallytrustedcallers metro attribute
fix selecttokens for empty containers
fix return type for sbyte and sbytenullable in toobject
fix a bug where certain json path expressions would spuriously fail
fix portable class library build
update reflectionutils cs removeassemblydetails
fix using a tracewriter with nullable bytes
asserted variable in writeescapedjavascriptstring is not null
refactor jsonserializer getconverter
add test with escaped json path
add jsontextreader read tests for empty stream
updated unit test to check for specific date value
add additional unicode encoding test
patch use of jsoncontract onxxxx to use onxxxxmethods
optimise linqbridge enumerable cast for direct cast available
don t call jsonwriter close when not disposing
adds missing directive
simplify equals override in jvalue
move execcompletes_issue943 to long running
lower it a tad more
what s a baby yak called i must look this up
fix iserver hang command in redisserver database number is not needed for client pause
don t set any socket options on unix sockets
save all the files
error messages on cast failures for redisvalue
rethrow any other
redisvalue hasvalue property
tests fix manythreads db check
and disable parallelization
link to the original author s code
fix test that is impacted by other usage
set socketclosed on endofstreamexception
add timestamps to test output
fix if calls close or dispose many times it ll throws a nullreferenceexception
revert misleading change to documentation
and another fixup
fix a bug in porting
tests cleanup on expiry tests
make usesharedconnection available in tesstconfig json defaults to true
no longer useful
tests more logging on scans
don t penalize us for reporting more than one way of failing
should have included tp stats in the sleep loop exit
recognize connection resets by name
fix build error caused by trygetazureroleinstanceidnothrow test
avoid race condition when add servers
missing some forawait
fix documentation for configurationoptions proxy
tests remove cluster spam
remove role from the isadmin group fix
keyrename return boolean
merge pull request from deepakverma master
for noauth authentication required recordconnection failure
tests move around
add missing import
trap the log in deslavegoestoprimary
tests fixes and debug info
when you heartbeat tell me why you think you need it
fix race in reconfigureasync snapshot creation
add clientname to output for timeout messages
default ports test was screwy
tests skip private ambient failures that aren t actually causing trouble
cleanup error event args
merge pull request from wangzq bug azure roleenvironment isavailable
remove unnecessary indirection to mitigate deadlock fixes
fix release tests
merge pull request from joncole master
put the azure cert logging into antoher test cos it be trolling me
clean up some compiler warnings
merge pull request from johnnycardy master
make sure that _writetickcount is set before it gets in the queue otherwise heartbeat will freak
cursors better error message
tweak to perf test rig
merge pull request from deepakverma master
merge pull request from deepakverma serverselectionstrategy
tests add output to profiling simple
add more indirection for the numerics check because il
fix a bug in the gettypecode extension method for non coreclr
prevent eval to script conversion if it is not supported for example twemproxy
add condition of dnxcore50 only check completed result
more case insensitivity redisearch
fix a if range on endconnect
fix off by one in the test
why because floating point path in linux is exactly what you d expect
remove reconfig call for slaving test
well since xunit throws that s a bad idea let s take another approach
tests fix preserveasyncorder_setconnectionmultiplexerproperty
another flow suppression
tests give scanresume a pattern
document default connecttimeout publicly
better test file
put the new warning later
iredisasync multiplexer iconnectionmultiplexer
sentinel get master addr by name throws exception when service name is not being watched by sentinel
tests fix similar dns issue for socketfailureerror
merge pull request from matteobaglini master
fix release build
use the new concatenate feature when add redisvalue to rediskey
deny exec sync explicitly null the delegates if it throws
just giving it a go
don t peek twice
check the endpoints are actually equal to populate the cluster master and slave nodes correctly
merge pull request from regiuskornel master
allow auth on twemproxy fixes
another log in the wait loop
disable stop words add
let s not nuke the snapshot field
the sharedconnectionfixture should just log ambient exceptions the testbase already checks them
fix resolvedns documentation
call setenqueued before the write lock
fix cluster simpleprofiling
re debug factory tests that are inconsistent
fix compilation using normal solution under mono
fix clientname summary doc typo
merge pull request from aaronmell sentinelfixes
prevent complete from being called twice on ping in a disconnect scenario
fix avoid nre and ensure that createbridge cmd can spawn connections if needed
use the new concatenate method
fix writing of message forgot to increment offset
don t append password if there is no password set it is just noise
tests time standardize format to include ms
fix debug build
issue path in comment for no good reason
testpublishwithsubscribers be more patient
use the supplied flags when selecting a server to subscribe to
tickle setauthsuspect if authenticateasclient fails
not all messagefaulted have an exception
nredissearch don t throw for everyone who doesn t have the module
on connection fail consider recreating it with retry
consider synctimeout when forming default connecttimeout
tests fix names exclude synccompletion to debug for now
channelmessagequeue tests await completion in others
wrap pipeline failures in recordconnectionfailed
update lastreadtickcount in readfrompipe
fix a bug where readonly is run before auth causes moved errors when reconnecting to clustered slaves
unhandled exception in callback when connection is disposed
subscribetoserver handle inernal call correctly
remove debug stopwatch
make georadiusresult constructor public to make it mockable
drop _serverencoder responses are overlapped now
appveyor is slow
fix false positive test
add auth to commandmap for sentinel servers
tests fix config oops
fix null ref in sentinel
streams fix pending message tests
get along tests
test exceptions add ya know useful data to this thing
debugging aids cleanup
updating to allow the creation of redisresults for mocking
log unaccepted results via messagefaulted
testpatternpubsub ping before and after the publish
corrected the bool operator comment
add gc info to test output
fix profiling test order check
tests log kinds seen in profiling
tests fix dns failures
fix flushfetchrandomkey harder p
fix my transgressions
taking care of remove extra if innerexception is not set
k server working i was just being daft still might as well keep this client id i add
tests dispose multiplexer in preserveasyncorder checks
tests guard text writes
try to fix appveyor messaging
streams fix naming tests
tests fix socketfailureerror for appveyor
booleans are hard
add test for
what the hell is that static doing there mri fmqo wqefoq w
remove usage of sslstream isencrypted as it is not availalbe on dnxcore on desktop it simply returns isauthenticated
fix physicalconnection beginreading exception handling
fix debug test build
fix version regexfix regex for previews
shorten test durations quite a bit half the pause
tests skip basicops testsevered
merge pull request from jeremymeng coreclr
tests bring stability to scanresume
becuase splleing is hrad
tests cluster getconfig better logging
oops those two methods are debug only
ensure message resultbox is nulled after successful async trycomplete
tests testmassivepublish debug info up top
woops broke debug
tests fix defaultclientname regression
merge pull request from derauk master
fix messaging about transactions vs batches
add server info and slot information to debug info
add new pluralization words for bias and iris to the default vocabulary
fixes the null exception on date humanize
merge pull request from gillardo master
add a hyphenate method which calls dasherize
remove unused parameter
replaced hardcoded resource keys with those from resourcekeys class
api documentation for dehumanization
remove unused using statement
add test to find dash hyphen with preceeding or following spaces or
titleize is now obsolete
remove redundant empty lines
updated the test to show the converter did not get the default numbertowordsconverter
add new pluralization rules for words ending in o that map to oes
just a field rename for consistency
remove unnecessary toarray conversion
correct ordinalizewithcultureoverridescurrentculture test parameters
numbertowordsextension towords add ability to convert billions
remove brackets for one liners
fix for defect bytesize trims number with format string
merge pull request from humanizr gloves
fix datetimehumanizealgorithms documentation comment
fix test monthsfromnow and yearsfromnow
merge pull request from codefromjordan master
fix missing format value for the argument exception
merge pull request from raathigesh master
remove testcase causing int conversion error
add tests to demonstrate defaults in
remove unused old factories
add test to reproduce
fix for re arranging inline data order
fix staff pluralization
return correct romanian string for hours ago
remove unneccessary trim from word length line
small refactoring for readonly instance
exclude from netcore
add new pluralization rules for some words ending sis that map to ses to the default vocabulary
re setting currentculture on ambient culture
fix assert equal argument order
fix bytesize not handling default formatting correctly
better name for unit test
add whitespace before if statement
add check for blank string
actually commit test
fix a review issue codebase convention
include water and waters as uncountable for english
update serbiannumbertowordsconverter cs
add a missing unit test for stringexentions
change modify portuguese ordinalizer
check that the pascalized word is not empty
add test for sorting a list
use simple cast
add cache caches as as irregular words fixes
merge pull request from anakic master
validate resource key before using formatwith extension
fix spelling error with forth fourth in englishnumbertowordsconverter
remove two test cases
fixes plus add singulars for other words of the same suffix
add tests for and fix a bug for sentence casing
merge pull request from ahmadalli dev
merge pull request from robpethick dev makelowercaselettersafternumbermatchpascalregex
remove brackets from one line conditions for code consistency
remove curly braces
renamed the tests to match the method names
use string interpolation beside string format
merge culture fixes from sveinfid norwegianlocalization
add string replace fix
cleaning up my braces
add readonly to ordinalizers
fix failing tests
add one more test case to check if preposition de is properly used
remove unused parameter
remove duplicate test for shoe
add human and freshman to pluralize test
add a test for getting culture specific resource with explicitly specified culture
argumentexception thrown if type arg is not an enum
add feminine gender for french ordinal words
add campuses as plural of campus
fix training pluralization
changed tabs back to spaces for consistency
use en us culture for apiapprover test to make the test pass on non english systems
merge pull request from philcarbone feature ifromattable bytesize
add broken test for
suppress code analysis message for idisposable in test
remove unnecessary trim
switch check to ui culture instead of culture
modified pascalcasewordpartsregex to accept more word characters
fix for defect minor re factoring
pass through the content of the letter other unicode category
isdefined is prefered
retrieve enum case name only once
humans not humen
neutralresourceslanguage resolves in windows store apps
disable cs1591 for bytesize class
remove multiple trim s within select add a single trim in aggregate
remove redundant space
remove test for genus
add rule for is are
add test to reproduce issue
fix miss inlinedata
fix convert signature in polishnumbertowordsconverter
fix test so words not at start of sentence aren t capitalised
changed the numbertowords converter to find the right converter by the language name aswell
add irregular plural for criterion it is criteria
updated string representation for the number
merge pull request from hazzik ui culture
merge pull request from llifon master
make sure that snackbarmessagequeue has the same dispatcher as snackbar
ensured combobox is not closed when scrolling
icons update from azure pipeline
nuget stuff skip ci
correct spelling mistake
icons update from azure pipeline
icons update from azure pipeline
fix issue with validationassist fontsize pointing to the wrong property
icons update from azure pipeline
icons update from azure pipeline
icons update from azure pipeline
fix a minor bug in maindemo wpf
fix for issue rename opened to closed in the summary
icons update from azure pipeline
fix link to dialogviewmodel should now work
attempt to load source locally
call passwordboxonpasswordchanged after initialization
ripple tweak for
try again with mahapps package skip ci
demo drawer handles scroll bar better
icons update from azure pipeline
icons update from azure pipeline
fix resharper problem
small build fix
change messagebox to snackbar in chips xaml cs
merge pull request from keboo comboboxassistfix
revert update palettehelper cs
fix display of color code
fix a wrong awaiter configuration
icons update from azure pipeline
updating comment to reflect type change
fix exception that can occur if there are no visual children when trying to find the appropriate name scope
merge pull request from spdjudd master
icons update from azure pipeline
fix mahapps progress sample
ensure retrieved datacontext on dialog hosts gets picked up from sender
fix demo wpf sliders style source link not working
icons update from azure pipeline
icons update from azure pipeline
materialdesignmultifloatingactionpopupbox rtl bug fix
icons update from azure pipeline
fix clicks not registering on datagridtemplatecolumn
fix combobox popup placement
remove obsolete property i m not gonna go semver on this one it s been obsolete for a while
fix setlightdark fpr relative strings
make public again my bad fixes
fix small build error
fix listboxassist istoggle toggling disabled items
icons update from azure pipeline
popupbox mouse capture fix
fix usepopup getter setter
add null check to dialoghost showinternal
fix when specified date is out of the calendar range
fix mdresgen constants
disable animation if on battery
moved return true out of if
reintroduce smarthint visual states mark as obsolete
updated ver for mahapps build skip ci
icons update from azure pipeline
ickle code tidy skip ci
icons update from azure pipeline
proxy hint content based on combobox iseditable
merge pull request from llifon master
allow null dialog identifer
null reference bug fix fixes
remove the old secondaryaccent resource with updated resource
update colorpair cs
add converters namespace to exported namespace
illustrate culture info in wpf fixes
recognizesaccesskey dependency property
datagridassist minor qol improvements
correcting issue where the picker would close after the hour was selected
isdropdownopen false on cancelcelledit and commitcelledit
preserve tokens if vm code couldn t be restored
allow zero target switch instructions
prevent stack overflow in check
update detection of proxy fixer
check for methods with no body
should ignore cctor methods since ctor is never static
fix version detection
remove call since value is cached
update help message
don t decrypt already decrypted resources
remove all anti reflection types
make sure enum instance field has proper flags set make peverify happy
update detection of winforms field names
dump embedded assemblies before decrypting methods
should also check for newobj instrs
initialize assemblyinfos in case there s no embedded assemblies
remove the dynocode declaring types
also use ldfld ldflda to detect arg types
update default regex
use the correct variable
cryptoobfuscator improvements for resourcedecrypter and antidebugger
use generic prop creator if the type has a generic parameter
cryptoobfuscator fix an issue on getproxycreatemethod not find delegate creator and proxy call fixer is not activated
remove unused assembly field
reset module name when reloading from byte
warn if unpacked data could not be loaded
deobfuscate string decrypter cctor
remove unused methods and inline method used only by sa code
add find2 method for derived classes
remove encoding arg
use a better method dictionary
make method static
ignore invalid method indexes
ignore invalid methods
fix method names
use new fieldtypes code
don t check whether method is virtual
don t throw if we can t find all method args in the same block
make sure no generic methods are inlined
update a few strings
update exception message
throw invalidmethodbody if ioexception
move key init code to a new method
add more asm search paths
change no control flow deob no cflow deob
remove old comment
remove modifiers before checking whether it s pinned
support calli instruction
add an inflate overload
method should not be public
rename class to streamxxx if it s a stream class
remove init method calls called from ctors
remove metadata token obfuscator type
move the field
remember to create dnr s info class
add inlinedallcalls property
merge pull request from yyjdelete master
add method to remove the assembly info
add peek method
handle call instrs with invalid metadata tokens
make sure method hasn t been remove
make sure initializecomponent detection code is called
update class comment
update valid name regex
also catch securityexception
ignore refs and defs from other modules
it s no longer beta
keep deobfuscating methods even if a cflow deobfuscation exception is thrown
update isbranchblock method
add another method overload
add removeguaranteeddeadblock method
return an szarraysig not an arraysig
add lookup method for modulereferences
proxy calls can be proxied
use the constant
add data property
add workaround for dnr patch bug
restore resource names ending in g resources
fix problem when haspinvokeinfo true but pinvokeinfo null
only call patcher if we need to patch it
fix for the last netreactor
remove non used code and add a count property
add some checks
no need to check for typedef since it s checked elsewhere
don t print message since the code is now much faster
update rename regex
check for null methods
set default key size
update string decrypter detector
add missing space
rename dr dump embedded dr embedded
don t treat system void as a value type
fix the last bracket problems
remove string decrypter type and allow static dynamic decryption
code should return true
remove encrypted resources from output file
make sure only valid methods are restored
this shouldn t be a warning
remove useless property
use the original module name if available
print a warning and use default encoding if the code page doesn t exist
add fileheaderoffset property
don t use iterator for speed
declaring type is null if it s already been remove
update invalid name regex
add methodcallinliner getfirstinstruction
fix problem when resources aren t encrypted or compressed
return number of method calls that were replaced
update error message
methods should be static and have a body
add readint32 and readbytes methods
check each part of the namespace instead of all of it at once
index should be set to instruction before we broke out of the loop
make method static
update detection of key
methods decrypter method could be null
fix bug in methods decrypter
add issystemobject method
ignore opt req modifiers
fix what was updated in master
remove castclass system string if present
print a warning if the file isn t a net pe file
print number of restored methods
let s use this variable
remove useless using
remove detection of babel in unknown obfuscator
add emulate method
add fileheader property
update the code
add isdelegate method
change type from ez to ef
remove assembly if one file option is used
print a message after unpacking a native file
write a message if decryption probably failed
add removemodule string and clearall methods
fix if condition
get rid of dead code
remove all invalid methods
assemblyresolver can now remove old unused assemblies
fix tostring should separate generic args with commas
make findsection public
add option only cflow deob for only deobfuscating control flow
check for null
add getextension method
add one more test
update valid name regex
add reload method when the file has been decrypted
add another instructionemulator initialize method
use the new name of this obfuscator
update df version attribute parsing
add missing null check
remove confusedbyattribute type
remove non referenced method
add option to disable method inlining
only print found native code warning once
fix forwardscanorder for tryhandlerblock
add getinitializeduint32array method
fix dotfuscator rename regex
update detection code
update dnr detection
update the counter
rename asmpath option to asm path
add another getfield overload
add fileheader and optionalheader props
set new locals by calling setlocals not by writing to the field
allow null input
add real8value tostring
detect another obfuscator
make sure decl type exists
remove public add static
remove unused method
make delegatecreatormethods list protected
update method name
update valid name regex
func should take a methoddefinition as first arg
update decrypter detection code
resource names are in strings so can t contain embedded nuls
add another warning message
don t call getdirectoryname if name is loaded from byte
use nop if an invalid opcode is found
make sure lastoffset filedata length could be a bad dump
don t restore method bodies from outside types
add call to stringdecryptersadded
add comparefieldreferenceanddeclaringtype method
don t add a base type if the type is actually system object
make method accessible by sub classes
don t deobfuscate cflow unless the method sig is void name
add better detection of our base dir
remove unreachable code
remove resources with an invalid rva
enable failing test
change confuser type from cn to cr
add log message that we have decrypted a resource
change parameters type from to ilist
mc actually does rename symbols so add an updated regex
restore bool char array ldelem stelem instructions
retry decrypting methods if it fails for some reason
ignore any exceptions during deobfuscation
update detection of clisecurert type
return if invalid index
also preserve blob offsets when preserving md tokens
don t email me when new versions come out
use ipeimage findwin32resourcedata
add decrypt method
if an ioexception is thrown it s not a pe file
don t remove any types methods etc if it s an unknown obfuscator
fix eazfuscator detection
add int64method property
create a unique metadata token since renamer depends on it
add isemptyobfuscated method
check for generic params in all generic arguments too
add a comment
ignore argumentoutofrangeexception when loading files
don t print error since the logger also adds it
merge pull request from etmatrix cryptoobfuscatortest
don t remove string decrypter types if there was an error decrypting strings
update error message
remove unused method
copy foundsig field
make sure info is copied
also check ldftn opcodes
master was updated
update version detection
fix serialization problem when calling exit
add hasinteger method
set security level to full
make sure we only decrypt resources once
make sure method operand isn t null
prevent merge if it s this
update encryptedresource cs
update detection of resource resolver class
print warning if i o exception
show obfuscator version
add another getmethod for generics
remove old code
update detection of compressor modulus
use a using statement to make sure the file is closed when we return
add rip comment
only remove the type if we rename types
fix the exception
add initializeddatacreator field to base class
update detection of compressor version
make sure user tries latest version
increment errors if there s an exception
remove encrypted strings field type
make method public
resource must be returned
assume invalid code so check for null
print name of encrypted strings resource
don t need to detect go here anymore
allow passing dumped methods to reload
also make sure type name is empty
updated log text
use new getinitializeduint32array method
add a new random name regex
add method to re read all blocks
print a message if methoddata isn t encrypted
fix name of method
don t return main asm if it hasn t been unpacked
don t try to inline methods without a body or no instrs
update printing of version number
use implmap ispinvokemethod
should be instance variables
return null if inflated data isn t an mz file
new files default name is now origname cleaned ext
remove unused variable
remove more xc attributes
print new resource name
update detection of that type
add sections property
update detection of compressor
don t cast the arg to its correct type
make sure rvas list is never null
update name regex
use a stable sort
call dispose after decrypting methods
add null check
should be continue
also add all methods found in vtablefixups
update code since getuserstring arg is now a token
inline the remaining methods
don t ignore mixed mode assemblies
update reason string
update if expr it was wrong
brtrue brfalse now checks whether it s a non null value
update detection of invalid cv methods
update text when reloading an assembly
also check for xaml extension
remove useless field
add method to find resource from strings in code
undo dup fix
ignore assemblies that contain native code
update program cs
update ru option text
update dnr name regex
update name regex
make sure field method ref has a declaring type before resolving it
use the setmethod property
fix exception messages
add entityindex constructor with equalitycomparer
fix destroy compile time error
printing entity in entityisnotdestroyedexception
should fix travis ci
add codegenerator to default searchpaths
fix generated code for repo flag
instantiating all hashset entity with entityequalitycomparer comparer
fix wrong gui color for contexts with retained entities
switch context is now minibutton
pool documentation maintainance
simplified obsolete check
pr cleanup closes
add createsystem type extension
use tostring on subclassed components too
fix logging wrong system stats
changed code generator cli keys and remove default values
remove gamestate from default contexts
merge pull request from idnoise fix indexed properties
fix flag components increasing the componentpool stack closes
remove cleargroups references
updated blueprints project
improved search field placement in dictionarytypedrawer
updated blueprints data provider and code generator to use isenabledbydefault
assemblyresolver won t append dll to exe extension
updated migration descriptions
groupobserver retains entities only once
fix test description
remove trailing slash from safe dir
add convenience ctor to jobsystem to use all available threads on the device
setting data sethasmultiple for custom entityindex too
remove info box in visualdebuggingpreferencesdrawer
add partial keyword to componententityapiinterfacegenerator closes
refreshing preferences when neccessary
using removedots method
add test for entity owners
fix generated code spelling mistake closes
shorter namespaces test message
add removecomponentwhenempty to optionally remove or keep empty listener component
ignore indexed properties in component generation and entity drawer
fix missing renames in entitasstats and systemsdebugeditor
propper whitespace in updated csproj file
add try catch block in contextobservergenerator for generated code to support unit tests
updated myflagcomponent to use propper flags
ignoring a test to pass travis
add window title to entitas preferences window
fix pool not shown when entering play mode while a blueprint was selected in the project view
stringtypedrawer now uses editorguilayout delayedtextfield
drawing member name and type
remove blueprints plugin from default properties
remove override debugsystems destroyallentities
add release button
displaying more prominent popup when trying to generate with compile errors closes
printing version and config in doctor command v
drawing public fields of unsupported types
contextgenerator sets aerc in generated context
add fast entities count call to pool
entity applyblueprint returns entity
fix running test within ide
typetocompilablestringextension now supports void
merge pull request from sdarkwell develop
fix switching between multiple properties
changed default blueprint creation location
merge pull request from bddckr fix component tostring usage
search string matches lower case
using word wrap label when there s an exception in the preferenceswindow
updated entitaspreferencesdrawer optimizations buttons
remove entitysetextension first
marked workaround for drawing header texture
moved componentinfo back into entitydrawer
add config key to ignore namespaces
support visual debugging of enum masks
merge pull request from trumpets feature contexts reset
systems list unfolded by default
catching exceptions when loading types from assemblies
updated old matcher class name
changed code generator keys and remove default values
disabled tests that fail on travis
fix missingreferenceexception occurring occasionally when stopping game
add and refactored pool tests again
updated namespace tests
add keepalive description to cli
even better names
fix pool createsystem not creating a reactivesystem for igroupobserversystem
improved entitybehaviour significantly by caching gameobject name
remove obsolete code
using typereflectioncodegenerator in unitycodegenerator
made all mehtods in systems virtual
generating different hashes for same sum of component indices for entitymatcher
using assemblypath in data providers
replaced entitycollection isempty with count
pool createentity and pool destroyentity are now virtual
blueprints findallblueprints orders all blueprints by name
changed attributeusage order
enumerating group instead of getentities
merge pull request from sschmid feature entityindex equality comparer
updated entitas unity to use entitas dll
tweaked debugsystemsinspector to be be aware of light dark skin
add context migration support
add icodegeneratordataprovider isenabledbydefault
simplified entityindexgenerator getkey
fix needing to generate code twice to when event got remove closes
fix not ignoring feature class in system stats
remove warning entitas abstractcompoundmatcher overrides object equals object but does not override object gethashcode
using entityindex constant
add creationindex to entity tostring
remove unused usings
reactive systems take ilist instead of ienumerable
moved groupeventtype to group
add getsinglexyzentity to code generator
fix hashset changes didn t replace component
fix code generator line endings for header
releasing entities in group when pool cleargroups
jobsystem execute is now virtual
rename entitiy to entity
entity documentation maintainance
merge pull request from bddckr fix entitiy spelling
add clear groups button
remove empty item groups in updated csproj
remove click logic from entitas header in preferences
add docs for pools creategroupobserver
more explicit entitylink tostring
import codegenerator namespace into entitydrawer
binaryblueprintinspector creates new pools instead of using one of pools allpools
changed warning color
renamed menu item
less editor repaints for debugsystemsinspector to improve performance
add comment to entity destroy
remove redundant private keyword
remove unused code
hiding component member searchfield when folded
wip still working on nested routing
add using to the cancellationtokensource
small grammar fix
customize the link generation at the markdown editor
fix the folder name case sensitive
fix broken build
fix the problem with the silo
remove unused code
reduce the sample even further
complete conversion to uikit
remove extra code
add beter comment
tighten the example
make the sample clearer
remove exceptionmessage property and use details instead
remove useless line of code
remove unecessary in
clean up code
change from async call to sync
json handling implementation
update program cs
remove unused using
change the resolver to use camel case exclude null and disable private
extend example to show inject at invoke as wel
show implicit struct conversion
demonstrate creating and deleting cookie
clean up code
use httpstatus movedpermanently enum
add missing using
remote unecessary code
do yet another attempt to make this work
use span of initial string
fix a minor typing error
specify hintpath for all referenced assemblies for mono support
respect size number format in metriccolumn fixes
additional info about windowsbrandversions
introduce general catch with hopes to help with
fix the unit test that i broke when i was fix
merge pull request from pentp master
params float double decimal support fix
fix currentruntimeisproperlyrecognized for netframework case on linux macos
improved legend for memorydiagnoser
don t print parse errors to the output fixes
remove from the processorname
make benchmarkrunnercore public
merge pull request from morgan kn sync
don t assume that targettype has reference to benchmarkdotnet fixes
rplotexporter findinpath handle exceptions trim quotes
remove old results otherwise the file will be overwritten and remaining old content might stay untouched
fix some spelling mistakes
merge pull request from vedun z master
show jitmodules in envinfo only for the classic toolchain
remove construction with description in benchmarkattribute
now runurl can work with non raw github and gist urls fixes
fix nre in benchmarkrunner runurl fixes
merge pull request from arthrp master
ensure we close the code section in the markdown we generate
net core toolchain root folder detection bug fix
update introbasic cs
we restore before build so build does need to restore too it s new default dotnet cli behavior part of
merge pull request from dfederm dfederm fix typo
disabletestparallelization for classic integration tests to avoid races
fix globalcleanupattributetest globalcleanupmethodrunstest
ensure we sort the results by method name description
addition to using single variable instead of two
increase miniterationtime accuracy improvement
fix namespace for jobtests
fix typo in baselinescaledcolumn cs
if the user provide a custom value we should use it post
implement frequencytests parsetest
update message in jitoptimizationsvalidator
distinct jobs to avoid possible duplication when config is doubled
ability to pass multiple assemblies
skip test that fails for classic on appveyor
always print the path args and working dir of the auto generated executable cc jorive
resolving jobid in defaultcharacteristicpresenter fix
lazy hostenvironmentinfo osversion fix in toformattedstring
better error message
add missing simplejobattribute ctor fixes
invalid arg passing in streamlogger constructor
trimming the argument values makes them actually shorter cc ahsonkhan
fix in hardwaretimerkind
shortify memorydiagnoser column titles
core dotnet cli supports only x64 now workaround
fix stupid bug with condition for optional mediancolumn
fix dots at the end not used in other places
enable default analysers in benchmarktestexecutor see
make fullnameprovider public so it can be reused by the dotnet performance repository
copy usersecrets from benchmark project
remove final traces or runtime and source sourcediagnosers
update array_heapallocvsstackalloc cs
print full information about a generateexception
always print full information about non optimized dependencies
flush custom loggers at the end of benchmark session fix
hardware counters don t try to exclude non existing overhead for long running benchmarks
params enum support fix
copying validators when merging configs fixes
use ascii mode for measurement presentation in terminal
allow using newer versions of microsoft netcore app fixes
fix typo in file_streamvsmemorymappervewstream cs
fix bug in baselinecustomanalyzer
fix event wire up before session acquisition
lazty hostenvironmentinfo osversion
add job core to command line arguments fixes
make logging robust when strings contain or
fix another problem in zeromeasurementanalyser see
fix compilation after merge
mark simplejobattribute with allowmultiple
missed out of last commit
benchmarkswitcher update the promt message
inclusive confidenceinterval contains
factorialwithtailing fix incorrect sample implementation
road to dnx fix tests
be more defensive when trying to read source code with disassembler part of
kaby lake r and coffee lake support in processorbrandstringhelper
if we fail to do the full build we try with no dependencies
check os version in windowsclock fix
use noinlining for commonexporterapprovaltests
processhelper handle null data values in outputdatareceived fix
support of benchmark methods with an interface as a return type fix
display result path in rplotexporter
persist optimized auto generated dll compiled from url plain code fixes
warn user if no columns were defined fixes
give users the anycpu hint when they struggle with badimageformatexception fixes
fix false allocations detection
remove dryjob from introbasic
fix sizeunit presentation fixes
fix multipleruntimestest singlebenchmarkcanbeexecutedformultpleruntimes
reverted reporter change
improved name for the id characteristiccolumn
add empty line before legends
support empty lines in disassembler getsmartprefix fix
fix compiler warning connected to problem with the xml tag
make sure that we can pass undefined enum values and the bug never comes back
spelling mistake in errors message see
withnuget should accept nugetreferencelist not ireadonlycollection nugetreference fixes
fix warning in roslyntoolchain
fix a bug where converting a bool to int instead to byte which is the native size
thread safe jobs fixes
update environmenttickcount hack
make runtimeinformation static internal fix
configuration fix in jitoptimizationstests
fail when running benchmarks in debug with defaultconfig fixes
add system type support in params fixes
better message in miniterationtimeanalyser
remove internalsvisibleto for samples application
update link to manual with net framework version constants
merge pull request from epeshk frameworkversion
skip test that fails for core on appveyor
ryujit is always avaiable for net core
wrong xml doc comment
don t log false alarms fixes
better mvalue formatting in multimodaldistributionanalyzer
update message to not suggest an obsolete api
generate the projects in the bin folder not solution s root
stop using binaries option which started producing new folder for every referenced project
add missed clean in benchmarkcompetitiontask fix
copy usewindowsforms too thanks onovotny for pointing this out post
enabled gcmodetests canenableservergcmode fixes
host mono process should be able to build net core child process fixes
detect net core benchmark failures from linqpad
fix order of exporters in exporterdependencytests
move sample to correct namespace
fix jitoptimizationsvalidatorismandatorybydefault in debug mode
it should have never happened but it does when debugging directorynotfoundex when trying to delete dir after calling exists
brandstring support for macos mojave
merge pull request from stevedesmond ca missspelling
remove obsolete namespaces in integrationtests
fix toolchaintest customtoolchainsaresupported
more tests for foldernamehelper
use hasvalue instead of casting
parsing gulftown processor brand strings
fix bug in op s reporting
newline fix in outputlogger
print full architecture name
speed up the integration tests closes
catch invalidoperationexception in classicbuilder
fix runalljoined fixes
fix type of local in emitinvokemultiplebody
road to dnx handling nuget like package versions that can contain text assembly version contains only numbers
warn the users when code optimization was not enabled debug build can be optimized
moving keepbenchmarkfiles merge logic to add method
fix typo in log message
detect situation when users want to use hardware counters with inprocesstoolchain and warn them fixes
don t fail with exception if user has written something to output in globalcleanup when using memorydiagnoser fixes
shame on me for not building the code for all tfms before pushing
throw exception about private benchmark method
use median instead of mean for overhead calculations fixes
change access modifier to private was not intended to be public
tests that ensure that when a benchmark throws the runner does not throw fixes
parallel build post fix don t write the compilation errors to nulllogger return the details which get printed afterwards
platformnotsupportedexception when reading processoraffinity on macos fixes
ignore auto generated files cleanup errors
add more time to lock unit test
update aggregate cs
fix shrink lock page
initialsize option is not supported for encrypted datafiles
fix unit test using int id for fix order
add cachesize parameter
improved json serialization in strings
merge pull request from lbnascimento master
use temp path in mobile tests
fix a few typos in exception messages
fix update shell command
add sbyte to mapper
fix again collection index name length
mini fix for last test oops
bugfix for sort with no enumerable
if filestream lock is not supported show an error message about mode exclusive
fix ensure in base page
made read only properties serializable
avoid dbref from null item array
fix unit test filename
remove raw_id method in expression
fix log file being created incorrectly when using encryption
change factory startnew to task run
support array expression resolver
fix findall in filestorage
add document scope expression
add dispose override
make bsonexpression explicitly sealed
remove enclosing in case of directly using connectionstring as file address
remove readtoend in person unit test
add more couner in shrink unit test
fix mscorlib exception which arises from journalling not being enabled
merge pull request from trueromanus master
merge pull request from lbnascimento master
fix disk_read_write unit test
shell unescape unicode characters before output
fix expression mapping for nested map calls
fix empty password being considered valid
merge pull request from lbnascimento master
fix upgrade database when has no file
use thread pool in diskwriterqueue
fix queries returning incorrect results
use utcnow in the filediskservice tryexec
fix litequeryable getplan not releasing transaction
fix snapshot creation when addifnotexists true is created after addifnotexists false
fix try catch thanks to rhullah
updating index count on ensure index
fix bson deserialization for doubles in scientific notation
fix double release transaction in query error
add check for do not accept index using not immutable method
support for null collection name in dbref use resolvecollectionname
remove unsed dump help
merge pull request from lbnascimento master
drop collection impl
remove stats from shell help
merge pull request from lbnascimento master
fix count func tsource bool mapping
add engine version to collection
bug log level isn t set from connection string
change repository firstbyid to singlebyid
fix abandonedmutexexception when a process crashes in shared mode
fix unit test for expressions
add keys function in expression
clear cache after transaction completion
optimize restore index avoid basepage class
avoid try recovery when journal is empty
fix clear cache sequence
better expression dot read
skip parallel unit test not fix yet
remove regex regex are too slow
assembly is now versionless
merge pull request from lilasquared serializedbrefid
fix missing check new pages on rollback
update assemblyinfo cs
add values document method
avoid debug window thow exception
add support for file storage migration
fix expr creation for nodes with expressiontype invoke
reduce exception throw when use complex linq query
add linq to bsonexpression conversion for equals method
fix mutex not being released after ensureindex in shared mode
merge pull request from disshishkov master
fix parameter order
making public getentitymapper from bsonmapper
minor name changes in
fix forupdate on deletemany query
fix parseupdate name
readded public to aesstream and tempstream
wait for task block task to finish
fix auto checkpoint
skip more unit tests
merge pull request from mstum setloglevelfromconnectionstring
add back id value in litecollection insert
skip unit test about collation thread
check value before executing contains fixes
fix connection string parser
rename executeexplainplan to explainplan
do not wait for exclusive lock in shutdown checkpoint just force quit
update assemblyinfo cs
merge pull request from axelgenus master
fix fileservice file
fix linq to bsonexpression mapping for nested linq expressions with same parameter name
require after map filter functions
fix concurrent unit test
change dictionary access
looking for page buffer release problem
fix stack connection when begin already runned
add sleeper to try avoid fail appveyor
add more shell unit test
fix query when has no collection but uses an source expression
create auto index with default options only
merge pull request from negue fix tests
fix nullreferenceexception when using filediskservice during shrink
fix limit and offset query parameters being overriden with default values in find method call
fix detect old verion shell
need fix setdirty order
fix nullreferenceexception in bsonvalue gethashcode for bsonvalue null
fix bsonexpression spaces on document definition
implement file limit size when request for new page
merge pull request from frosthage master
add dbref static method to bsonvalue
fix duplicate _id mapping
add linq to bsonexpression conversion for datetime touniversaltime
merge pull request from lbnascimento master
fix collection sequence in shrink
fix connection string mode
studio check middle button tab
better error message in local drive
fix nextindex dict extension
add tobsonarray extensions method to ienumerable
bug fix in indexin
merge pull request from petersondrew webp mime type
remove keyvaluepair test
fix includes not being passed from litecollection to litequeryable
fix parseupdate name
remove extra log
new explain plain way
add more shortcuts to literepository
fix incompatabilities with net core
merge pull request from olehfb master
return sharedcount in ensure page destructor
fix last binary page
fix missing select to full document return
merge pull request from falahati master
add keyonly option on select method
update ioexceptionextensions cs
add methods validations
fix repository action query
change dispose liteengine to rollback
add more examples to implicit test
use mapper serialize method for dbref id
disable broken benchmark
remove dispose in destructor
fix null reference exception if db file isn t open
support filestorage ids with spaces
merge pull request include utc offset in log timestamp
fix console logging typo in mirrorprovider
remove unused using
merge pull request from fix failure in full ft
update initial connections and wait time
back to upper case since it works fine and keeps it consistent
don t allow switching between anonymous authenticated and treat empty creds as error
merge pull request postfetchstep do not immediately delete expired commit graph files
use explicit string comparision
substituting a var with explicit variable type
upgrader write out more informative error message when install fails
don t use net core specific remove
revert whitespace change
merge pull request productupgradetimer remove unacceptable environment exit
add manpage link to comment
fix stylecop issues
gitclenupstep fix race condition
ensure git credential cmds are run with correct wrkdir
update includeverb helptext and output for list when there are none
fix stylecop issue
add subdirectoryinfo object
little changes from pr feedback
incorporating review comments
merge pull request gitprocess catch invalidoperationexception when setting priority class
fix highestversionavailable traching in tryquerynewestversion
use backing root in mount core hooks func test
add method to remove all entries for a folder
commitgraphstep do not immediately delete expired commit graph files
add sparse prune message when finding folder to prune
switch from fileattributerecallonopen to fileattributerecallondataaccess
add method to dehydrate a folder
update error message in postindexchanged to match check
use bashrunner for deleting the backup directory of dehydrate tests
fix diagnose message for cache server
add single path arg version of getbackingpathto
force checking out the commit used for the file listing
merge pull request add more time for disposal
upgrade upgrade help text for the noverify option
use generic posix wrapper name in func tests
exclude categories mactodo failsonbuildagent tests on mac
early exit if the named pipe is already disposed
quote paths when calling the command line
run post fetch job immediately on gvfs mount to ensure users have an up to date multi pack index file
fix stylecop issues
add more time for disposal
merge pull request from nickgra mirrorproviderreadonly
fix stylecop issues that snuck in after rebase and add some debugging statements
merge pull request upgradeverb tweak message logged after checking for update
remove extra assertion for expected error
tryprefetchcommitsandtrees downgraded to warning
fix mixing to make string interpolated
fix stylecop errors
disable native_projfs_movefile_partialtooutside functional test
gitprocess repack kept packs
reporegistry properly log error from trygetnormalizedpath
revert unintended change
merge pull request mac disable statustests movefileintodotgitdirectory
fix another windows build break
merge pull request gitprocess enable lower processpriorityclass
dont try to trigger reminder messaging when running in unattended mode
do not take gvfslock for git submodule commands
nugetupgrader add details to error message
fix path for the mac
merge pull request remove parameterization from gitblockcommandstests
fix mirror provider
fix bad test by fix maintenance step lock string
add more information to message when kext is loaded successfully
merge pull request from fix upgrade timer crash
merge pull request disable native_projfs_movefile_partialtooutside functional test
merge pull request turn on packfilemaintenancestep
use shouldbe syntax
merge pull request log ring when querying for upgrades on the background timer
add wait for background operations to complete in modified paths test
ignoring gitssl tests on non dotnet core targets
fix bash delete test failure
isfilesystemsupported cleanup test files on mac
use dotgvfsroot constant in place of gvfs path
rename property used when index is changed
allow downgrading git on install
check return code when trying to add new folder to include
productupgradetimer remove unacceptable environment exit
updated functional tests
relax looseobjectsteptests from exact object counts to positive or zero checks
fix full ft failure
handle folder created add to included folder for mac
use fulldehydrate variable in all places
tests demonstrating issue with newline in git commands
update message for failed folders during dehydrate
looseobjectsteptests unmount before running
gitprocess enable lower processpriorityclass
add restore and switch to list of known git commands
keep the existing placeholders hashset up to date for reexpansion
fix compilation error on windows
fix filesystemcallbacks bug in updateplaceholderifneeded unit test
merge pull request minor path comparison fixup in productupgrader
githubupgrader handle taskcanceledexception that indicates a network timeout
update comment for creategitscript
switch from giving everyone read access to users
add additional diagnostics to gitstatusafterrenamefileintorepo
add test for when modified paths gets deleted when it shouldn t
use isunderconstruction to avoid threadabort on mac
telemetry record when a mount succesfully starts
log ring when querying for upgrades on the background timer
force untracked files to be shown when running status for dehydrate and include verbs
upgradeverb tweak message logged after checking for update
mark moveindexfilefails as maconly because cmdrunner on windows is failing
handle read only files when hydrating with the mirror provider by only requesting read access
merge pull request upgrade run upgrader in system diretory
rephrase projfs error messaging
upgrader do not include micrsoft diagnostic tracing eventsource
update disklayoutversiontests to new version number
remove unused code
gitstatuscache log when cache is not enabled
merge pull request gvfsverb explicitly disable new config settings
merge pull request ensure git credential cmds are run with correct wrkdir
productupgrader use ordinalignorecase in exluded hashset
use datasources allbools in unit tests
simplify virt paths in dehydration status tests
use refresh rather than creating a new fileinfo
remove upgrade on minimum verison test
merge pull request fix console logging typo in mirrorprovider
renaming loadcertificate to getcertificate
fix addstagetests on windows
log filemode in eventmetadata
postfetchstep delete and rewrite commit graph on write error
merge pull request add wait for background operations to complete in modified paths test
shutdown gitstatuscache before components it depends on
move jsontracer code for writing to non failed listeners to a common
use legacy rebase
perfprofiling should not initialize gvfsplatform twice
folderdehydratepreviouslydeletedfolders add more validation
remove redundant dehydrate folder deletion test
getrelativepath does not exist in non dotnetcore replacing with my own implementation
using gvfs platform posix
remove windows installerextension
merge pull request getinsertionindex should include the entry s name when it throws an invalidoperationexception
merge pull request repack use one thread but allow deltas
merge pull request fix a flaky test involving the background prefetch thread
ignore test for build machines because wsl not enabled on them
remove unused method
fix failing functional tests
update comments and help text
merge pull request add new microsoft certificate thumbprint
add functional test the fails on windows without fix
fix eventmetadata construction in trydownloadprefetchpacks
set core filemode based on support
fix notification registration in mirrorprovider
increasing precision of timestamps being logged
do not attempt to approve credentials for anonymous access
align cmdrunner access denied messages with others
readmessage ioexception should be a warning
nugetupgrader tiny cleanup remove an unnecesary assignment
specific comment on case differing sparse mode dir
nugetupgrader fix typo in private method name
productupgrader be extra careful about a copy loop
fix net framework compilation issue
fix ut failure launching upgrade tool was moved out of the spinner ut was still expecting the succeeded message
mountmergeslocalpreposthooksconfig add more logging
fix test that moved new folder outside the repo
fixup bad merge
merge pull request run prune code even when no folders change
prefetch tests delete multi pack index before indexing a pack
fix upgrade timer crash
cleanup for pr comments
call allowindextobeparsed in the tryreleaseexternallock method
fix test the will no longer have child entries of a folder in the modified paths
ensure consistent ordering in sparse func tests
simplify logic for onhardlinkcreated
merge pull request fix issues that lead to problems with unmounting a repository
move code to remove modified path entries to the trystart method
fix dehydrate test
add missed file
add sparse option for disabling and returning to a full enlistment
delete any existing tools directory before creating a new one
incorporating review comments
merge pull request fix typo in serviceverb help text
merge pull request prefetch tests delete multi pack index before indexing a pack
hooks fix message around gvfs status
avoiding possible redundant initialisation of concurrentdictionary in filebaseddictionary class
gitauthentication remove credential string from trace data
fix a flaky test involving the background prefetch thread
update placeholders contains to check for null as well as dbnull value
remove unnecessary under construction check for connecting to the mount process s pipe
pr feedback to cleanup disposing and returning connections to the pool
run prune code even when no folder change
fix windows build error
merge pull request do not attempt to approve credentials for anonymous access
resetmixedtests drop test that behaves differently
got rid of unnecessary under construction check
mac disable statustests movefileintodotgitdirectory
patch up consumer of trygetfromconfig
fallback to slow method of getting process
powershellrunner list the path name that failed to hydrate
add tests for commands when files are hydrated
trivial fix a typo in error messages
add null check when creating trace metadata using and exception
merge pull request use which to locate git installation
productupgrader add empty constructor for mocking
clean up some code in gitindexprojection
merge pull request from rephrase projfs error messaging
fix failing functional test
fix upgrade tests that were broken
turn on packfilemaintenancestep
minor path comparison fixup in productupgrader
don t fall back to installing via inf when enabling the optional feature fails
fix string normalization in tests
bitmap file locks the file
check for null label feature and groupid columns
update user name in loadable class attribute to the right name
do not pass validation set and metrics callback to image classification api
loading the label for binary classfication as bl instead of r4
averaged perceptron pipeline serialization fix
polish some options xml document strings
allow only binarypredictiontransformer estimators in ova
fix a bug in columntypeextensions samesizeanditemtype
mark get next pipeline test as ignore for now
fix to enable calibration in ova
enabling the disabled tests
add xml doc to the itrainerestimator interface
remove dummy command
internalize microsoft ml data evaluators folder
internalize microsoft ml internal cpumath generalutils
add correct validation check
remove unnecessary jit workaround
remove subcomponent usage from ml pipelineinference
add culture independent parsing
silenced kmeans benchmarks otherwise it treats console output as errors
using invariance culture when converting to string
improve error message for non vector input to tensorflowtransform
improve vectortypeattribute dimensions docs
updated aucaggregator cs computeweightedauc method to fix spelling issue of defined in message
quick fix to xml documentation
add cancellation checkpoint in valuetokeymappingtransfomer
add automlservice test s kety to assembly
add document for method chnagemodelthreshold
printing additional metrics in cli for binary classification
fix build break
fix assignment of a variable to itself
contributing fix typo
updated the api call for getrowcursor as this was causing a build break
fix typo in the method summary
pass weighting value from producewordbags to wordbagestimator constructor
warn when a graph contains a null output
internalize tensorflow api and fix
exit when exception encountered
clarify model format in tensorflow model loading sample
updated with correct tags for image reference
fix grammar in exception message
api docs for experimentation
update textloader cs
add a test handler for appdomain unhandledexception
nestedobject for pipeline
fix documentation for lightgbm parameter
customgains should allow multiple values in argument attribute
featurizetext add instructions to turn off char or word gram generation to the tooltip
setting model directory to temp directory
fix the link
add internalsvisibleto to a couple of projects
conditionally compile helper code
fix build break
cli ifileinfo null exception fix
add internalsvisibleto in automl and codegenerator for the assembly microsoft ml modelbuilder automlservice gpu
bug in check for existence of group field
use random file name for automl experiment folder
updating the databaseloader to not force vector for single element columns
default initialization for ignore columns
fix quotes on json
link to an example on using converting ml net model to onnx
fix bounding checking error of sse sumu intrinsic
add cancellation checkpoint in normalizer transfomer
fix lightgbm ranking sample runtime exception
one line change replace a xref with cref in xml
fix typo in the featurecontributioncalculation sample
disabled autofitimageclassificationtraintest for test stability
fix new file
automl return null instead of null ref crash on model property accessor
fix the build
remnants from renaming of stratificationcolumn
enabling savepipeexponentialaverage test
fixes locale dependent test output comparisons
fix runtime exception in prior trainer sample and update the sample output in comments
always checks label in ffm trainer
fix documentation for imageclassificationmetricscallback to resolve the confusion in issue
make internal trainer constructors visible for specific projects in internal repo
fix copy paste error
fix onnx output for gcntransform
output memory usage after every test finish
typo fix duplicate in case of remove
use full test name
typo in comment shoudl changed to should
change the param name
remove unnecessary finalizer
add cancellation checkpoint in logistic regression
infercolumns api that consumes label column index only rename label column to label for headerless files
calculate reducesum row by row in onnx model from onevsalltrainer
make microsoft ml ensemble internals visible to nimbusml dotnetbridge
add gc keepalive for tensor variables
correct spelling of computeweightedauc
spec add additional checks
maint log that license file has been found
ifilesystem get_directories hardening
maint applicationparameters are readonly
maint correct whitespace
process adapter add waitforexit int milliseconds
add packages location to applicationparameters
maint ensure no null on enumerable iteration
maint remove comment
maint add todo
bug source user should use passed in username
new ensure the xml declaration is first line of file
maint print choco version when no args
maint use installshield installer when installshield
maint comment on updating source nupkg
log add more logging around pack
maint fix packageresult errors if downloadurl is null
remove container binding log initialization
maint licensing updates
maint fix message
maint clarify password is stored in config
doc note what cachelocation replaces
uninstall force deps aka remove deps
command accept features as alias
doc uninstall explain autouninstall setting
specs fix upgrade scenario specs
install remove nupkg from source arg
infra accept whatif what if for noop
maint provide better warning on non admin prompt
maint reorder specs
maint formatting order
install require answer to prompt
log success message important
maint comment out config in commands for now
maint remove message
maint remove comment
specs reduce retries on filesystem for specs
maint remove comments as they are in the todo
doc add returns statement
specs mocklogger wired up for specs
log improve message on upgrade error
maint log update message about number of tries
maint comment grammar
specs localonly logic moved closer to call
doc scripting how to get exit code w powershell
maint add todo
doc fix typo
doc move around command names
maint set application parameters readonly
maint add logging for lets
doc warning on verbose debug blended option
maint corrected whitespace
doc synchronizer enhancements to autouninstaller
mono no check on is admin windows specific
maint don t print chocolatey version every cmd
spec remove logger
maint skip log powershell resource assembly failures
list making list verbose more awesome
doc info add usage examples
maint move magic string to constants
specs remove logtoconsole logging
doc corrected typo
autouninstaller uses remove surrounding quotes
maint log option parse error w converted message
prefer assembly getexecutingassembly
maint test fix part tres
spec add expectedexceptionattribute override
log powershell contents w out formatting
add warnings to resulttype
maint remove user machine from main error
spec ensure mocklogger is setup first in test suite
maint todo note
config add verbose for list
maint rename environment path separator
pack the path must be in mono
maint some options could be achieved w foss
maint format_with should not error on null string
maint configsettingsservice provide noop msg
maint do not load config back into global
maint prefix nuget on nuget logger
maint fix misspelling
maint fix specs
event more to event messages
maint add local to the aliases for local only
specs bundled options are atomic
infra ifilesystem get_current_directory
log only log config warning if not limit output
doc explain upgrading licensed version
merge pull request from jbuedel stable
pkgsvc fault tolerance failure location
log explain how to install licensed extension
specs remove and readd install envvar
specs fix scenario for failed uninstall
log remove extraneous logging statement
specs start setting up host for executing specs
maint let the files get unpacked and wait one
maint variable naming
maint message on reasoning for using different config files
maint api warn on extraction errors
maint remove extra
posh add more environment variables
maint virtualize get_package_information
ioc bind ichocolateypackageinformationservice
command list v should pop on package name version
spec set uninstall base scenario
log replace_file use new lines for log
maint comment for webpiservice
specs fix pathing expectations with mono
maint remove spaces
pack ensure we pass the path so that mono works
maint fail on new file when not supported
bug do not change the exit code that the application has set
specs add more todos
maint template share location should be appropriate
log show exit code when running v
resharper ignore naming in file from library
maint msi install log parsing comment
maint add more information on file extraction error
maint remove sealed modifier from attribute
log update what confirm y does
maint move readme template to top level
maint update email address noted
maint fix logging message
maint add unknown type for platformtype
maint test fix redeaux
infra implement deep_copy
specs force unpack
config prompt for confirmation
maint do not log warning unless regular output
config save features to config file
doc change wording on name parameter
maint additional version environment variables
log log actual file ran instead of string
new make authors more clear
maint template specs restrict to windows only
spec doc mocklogger logmessagestoconsole usage
specs update todo
maint allow overrides
maint fix a choco version command help typo
spec tinyspec pending notworking
nugetservice upgrade noop means install noop
spec ensure trace messages are not logged
log add debug log search filter for nuget list
doc new parameter choco version availability
config add includeregistryprograms for list
specs allow for nupkgs to finish copying
template remove deprecated functions
react to aspnet routing
urlhelperfactory geturlhelper throws nullreferenceexception when passed
generate portable pdbs in coreclr
chaning unit test to use mock setup instead of setting it directly
set designtime to false
temporarily disabling running
fix how initialize is called
updating builderextensions to use the new named maproute overload
make razorviewengine getnormalizedroutevalue public
correct extension class names to match containing files
use datetimeoffset with fix time zone in tests to allow it to work in
merge pull request from sikebe fix locale sensitive unit tests
actually fix build break
use strict samesite mode for temp data cookies
react to aspnet razor
fix razor test compilation failure
fix functional test random failure
react to di changes
filter out generic invoke methods in viewcomponents
remove unnecessary call to metadataprovider getmodelexplorerfortype in controller tryvalidatemodel
update controller cs
call into the iactionresultfactory only if the return value isn t iactionresult
react to renaming targetframework to runtimeframework
fix for issue
fix the test to expect characters produced by new ihtmlencoder
normalize line endings in razorengine tests
perf remove extra modelstatedictionary allocations
remove unneeded internalsvisibleto
cache modelmetadata hashcode
remove double null check
remove usage of razorirnodevisitoroft
align taghelperoutputextensionstest casesensitivetaghelperattributecomparer with razor runtime version
remove isfrombody property from modelmetadata
remove workaround for aspnet external
fix merge issue
fix musicstore html beginform while generating the action url
disabling hanging flushpoint test
minor remove additional space from indentation of referenceassemblies hashset
react to logging changes
make sure references are readable multiple times
updated test to fix build break
use assembly s containing directory to locate precompiled views assembly
react to dnx renames
addressed feedback from rynowak
fix bad xml documentation
localizedhtmlstring value doesn t include arguments
more reactions to di
fix razor tag helpers pre compilation bug
fix for actionlink generating wrong urls
fix build break
temporarily disable clientruleswithcompareattribute_errormessageusesoverride
react to api change in httpabstractions
fix xml doc comment in applicationmodelconventionextensions cs
switch to use iservicecollection
disable broken anti forgery tests
merge pull request from coskunsunali dev
update razor precompilation diagnostics
react to changes in caching
responding to interface break
fix build break
fix wrong order of arguments passed to argumentexception ctor
fix a build break due to api change
fix tests broken by renames
re enabled the test
fix build break
add code path for rendering html encoded content on k
fixes re introduce httpmethodactionconstraint dictionary perf improvement
remove unused variable from roslyncompilationservice
react to roslyn changes
react to authorizationcontext rename
ignore line ending diff
fix obsolete constraint resolver usage
fix doc comments
bug remove addmvc overload with default parameter
treat returned strings as content results
fix nre in mutableobjectmodelbinder
a new pattern for add multi registration services
remove options order test cases
return a shared contract resolver
update message from di exception
linux fs is case sensitive
react to changes in antiforgery api
temporarily skipping failing test
fix build break due to aspnet configuration
fix xml docs build error
fixes add constructor taking the bindertype to modelbinderattribute
react to aspnet razor changes
react to aspnet razor
react to aspnet razor enable skipped test
react to errorpage middleware changes
react to razor changes
react to aspnet session changes
updating connegwebsite name in tests
react to iassemptyenvironment breaking change
reacting to diagnostics error page changes
nuke extra curly
update mvccoreservicecollectionextensions cs
add missing access modifier
fix unit test
xml doc fix
react to security logging change
fix the sample to await when writing directly to the output stream in a controller
run requirehttps filter early in pipeline
correct modelmetadata isrequired to honour overrides
updating tokenprovidertests to use getauthenticatedidentity instead of creating genericidentity
rect to razor rename
disable flaky test processasync_worksfornestedcachetaghelpers
change actionmethodexecuted to be verbose
remove redundant null logger after hosting fix
make getsupportedcontenttypes on inputformatter vitual
bamlmaptable should always reinitialize knownassemblyinforecord
set inbrowserhostedapp false always
better timespan comparison with zero
merge pull request from dotnet campus t lindexi fix xmlnsinfo comment typo
fix for relative uri s in flowdocument being broken
merge pull request from dotnet campus t lindexi fix doc
revert back to not using linq which ads unnecessary dll loads
update obsolete values of debuggerdisplay attributes
fix range of for statement
merge pull request from dotnet campus t lindexi replaceisassignable
remove bug id
update osversionhelper cs
merge pull request from andreyakinshin debuggerdisplay
overriding gethashcode for milcolorf
merge pull request from dotnet dev sambent datagridapfix
fix dynamicscriptobject cs
replace isassignablefrom to is pattern
merge pull request from dotnet user danzil nolinq
remove bug number
avoid allocation of string when a better stringbuilder overload exists
update the docs link
fix if no agents are connected to an external brain the external communicator should not try to update the agents
minor typo fix in monitor cs
check for potentially bad env variables
fix for the tennis agent works with prefab now
add note about y flip
merge pull request from unity technologies hotfix bw scaling
set maxstepreached to false on reset
make a few things in rayperceptionsensorcomponent public
checking if the past action is null before feeding into inference
update brainparametersdrawer cs
fix on the setcumulativereward method in agent cs
update timer strings
fix deleted observations
an agent can now spawn an agent in agentstep
add empty line at eof for decision cs
deregister the same method we registered
fix missing action axis in tennis
add an error message when no model was present for inference
fix typo in braineditor
disable compile warning
reverting bug introduced in
add a description to projectsettingsoverrides
fix bug in brainparameter drawer
fix comment in academy cs
updated the reacher s model file also updated the reacher s agent code from eulerangle to quaternion
update com unity ml agents runtime sensors idimensionpropertiessensor cs
use time captureframerate to ensure synchrony between update and fix update
fix cumulative reward reset
replaced collision enter with collision stay
fix unit test
remove allocateoutputs and outputfeatures
merge pull request from unity technologies hotfix no brain initialization
actionsegment isempty docstring
fix so we can now destroy the agent in agentondone
use mlagents sensor namespace
fix scaling of greyscale visual obsrrvations
don t make the test a monobehavior
crawlerlegcontact incorrectly refers to ground
fix compile error
fix batch size for visual observations barracuda was mostly affected by it
error message improved the error and add a link
merge pull request from unity technologies develop sensor access
use instead of dictionary add to avoid key exception
addcomponentmenu for decisionrequester
remove a debug message
fix broken code
update com unity ml agents runtime sensors idimensionpropertiessensor cs
handle refund echecks
set swagger host
tweaks to tokens
revert authority to internalsso
remove duplicate quotes
wrap file body stream in using block
check for keys already existing
rename policy types
custom plans with expiration are not trials
todo for payment failure email
call proper sproc
dockerhub version by first character number
check for settinghasvalue
log apple status update
edit bug on create
jobject fox for ciphers push
log to console is not quiet
add null check
format ban logs better
null check user
log from baseidentityclientservice for testing
remove test exception page
new device type return device ip info on events
owner check when editing
allowcredentials for cors policy
add back post verbs for backwards compat
remove unused using
performed some null checks
move all security headers to web vault location
cancel any subscriptions when deleting account
catch errors when trying to resolve dns
support for sources in webhooks
db maintenance function
remove old service refs
make cloudqueues readonly
contentdisposition attachment for azure blobs
push fixes and additions
user event route update
disable click tracking on plaintext emails
fix placeholder for formatting date
fix sdk reference issue in billing proj
prod licensing certs
enabled sso when needed when upgrading from a free plan
back to regular ip string
trycreate uri first
fix spec link
sync org keys notification when remove from org
allow long urls on admin
only apply count restriction to non deleted users
remove email verif check on license for test
proxy to admin
update error message
remove old actions
add server validation for plan selection
change to username
route template in http verb attr
credit payment type
fix null refs on free org create
correct cors policy for events
decompress gzipped icons
log when requests are blocked
check folders as well
check to make sure user actually needs key update
swap around sso service uris
url encode tokens
dev licensing cert
twofactorproviders only for two factor response
run jobs in local timezone
proper tocipher on cipherwithidrequestmodel
before running start
check user access permissions on collections edit
show pii on api
no extra new line needed
resolved circular dependency in user store
add basic constraints configuration for self signed certificates
larger fallback url for mobile
disable unused endpoints on identity server
org event route update
remove additional queries since they can be abused
serializer and swagger adjustments for dev
small logic fix
update ssl ciphers to mozilla recommendations
add revision date to folder and site responses
dont send pushes for now since they depend on user
allow blobs in object src csp
web policy for two factor apis
push apis are only for not self hosted instances
storagebytesremaining by way of users get premium
wrap csp in quotes
allow alive check on cloud an self host
linq optimization for android devices in push service
throw no payment method is available
only log cipher deletes where user can edit
add security stamp to profile
remove debugging true condition
self hosted org users get premium access
subscriptions are always canceled at end of period
add more gateway enums
update rolling file settings for logrollbysizelimit
ref event should be base64 encoded
tweak buffer sizes and filter error logs
version version header
add back billing settings
add missing redirecturis
fix admin cookie expiration
add key to login response from identity
env print updates
fix double hashing vault url
set usepolicies on license update
no need to owner group change check
putcleartoken requires authorization
allow custom size to be passed in
org name for mail url
disable autostats for cipher table
delete with etag
fix cipher too many args for sql
ef repo fixes
increase folder limits on imports
disable click tracking for text breaks all emails
filter id server tokenvalidator error logs
add stackexchange sites to equivalent domains
handle applereceipt already existing on customer
correct noop implementation
migrate script names for dbup
set filename with contentdisposition
comment on expiration
deprecate authcontroller in favor of identity server
globalsettings sso cache timeout setting in sec
log keepalive task
support x real ip
remove unused field
allow delete verb
add businessportal property for use on client side
tvp type name
add check type to payment methods
use sso external url
noop event service for now
fix organization put drops cipher updates
handle nameid as email w o email attribute
premium check for attachments
fix token refresh logic
increase timeout for user delete sproc
fix expanding folder bug for sites
replytoemail in env file
add internal api resource
turn off database autoclose
proration adjustments for next bill amount
map plantype to old strings for license hash
amounts are positive always
log httpclient send exceptions
check enabled too
issue license key and store expiration date
fix validation error for free org accounts
totp to login response
log errors with full exception
fix a broken test
add proxy prefix
fallback globe image
alive endpoint and logging
validate all installation ids
max additional users check
fix alive endpoint on notifications
remove error log on send
support for stripe sources in billing info
correct connection string for res queue
filter canceled orgs
we dont need to setpropertiesasync on upload
fix null or whitespace logic
allowed media types for icon images
uwp device type
add safari to download function
match default is null
default smtp ports
cipher details public
make sure user services are registered
parse cancellation dates to detect refunds
starting informational with bypass
more user friendly error message for no access
always map mssql logs
reduce new lines
set connection header for signalr hub
catch gateway exception when canceling on delete
pass down version properly to generate license
set nsubjectaltname od self signed certs
default user culture to en us
update app cache when editing orgs from admin
fix faulty conditional logic for showing enabled policy labels
update certbuilder cs
fix logger ref
merge pull request from bitwarden soft delete fix
remove noorg param
remove www url from alibaba
disable send for release
missed one other pr comment fix
port fix for setup
back to localhost
try explicitly setting authorizationendpoint
add text plain media type for all json input formatters
collect events in bulk
push controller is only when not self hosted
updated stripe plan id for family
remove premium checks for internal testing
return profileorganizationresponsemodel for organizations
matchtype proper match
re enable send
require sso add service layer dependent policy check
add max storage gb
empty ctor for note model
upper card brand
map yahoo login
new lines on warning
set owner context for org api
remove protocol from host
return after no master password hint email
log events on bulk delete ref
validate email on org create
subscription update to maintain auto charge
handle invoice already paid error
no need for chrome ext to have appid
use custom discoveryresponsegenerator on cloud too
fix credit card bug for premium
fix null ref
merge pull request from bitwarden fix free country required
register global settings
handle two factor remember failed case
no ssl warning
new line at end of env override files
check for org user email to be null
update location and state for generated certs
fix mssql connection string in setup
assign usepolicies when upgrading
xforwardedfor on self host
update billing only if something changes
rename method get getuser
add discord to global equivalent domain
remove unused service
revert https stuff
icons uri for web settings
import should be in the transaction scope
allow gravatar in csp
usedeveloperexceptionpage only on dev
add usebusinessportal to orguserorgdetails
if check on db migration of scriptnames
continue if disables
csp is only for web vault
return existing device for token
add authenticatioon to fix di errors
update plans with new stripe plan ids
null ref fix
do not require uri and password for sites
prevent duplicate invite accepts
bitpay invoice id
sqlconnectionstringbuilder to make connection string
smtp env config
requestsizelimit for all attachment endpoints
add missing enum type
adjust icon size query
suppress logs for bad security token
remove install complete message
fix logger category
fix null ref
null check response
only azure ban on non self host
allow updatekeyasync when user already has key
getlicense api is for only not self hosted
remove old org property
not null whoops
context middleware comes after auth
fix syntax error
no longer need hibp in connect csp
do not accept bank tokens for premium
update stripe plan names
mappeddomain is cache key
requestsizelimit for attachment uploads
return not support extension messages
fix ctor for stripe payment service test
fix info logger for apple iap
filter out heartbeat warning log
trim question input
remove console log
pass org admin flag for attachment actions
switch device type around to match mobile enums
stripe invoice number
return continuation token
swap dates if start end
new properties to profile
remove logging from base identity client service
use internal sso for metadata address
add another desktop ip
adjust match types
add bitwarden server category
map nginx logs volume
trim quotes from env file value
null cached entries are high priority
cancel on delete
revert larger fallback url for mobile
only get devices with push tokens
blacklist of link rels to ignore
normalize billing email
dont allow org id assignments on cipher creation
add azure to signalrserverbuilder
dont catch exception for testing
only fetch icons from http s with default ports
make sure the tray menus got translations
add empty actionparameters check
add reindex when updating plugin program suffix
add production logging
fix space issues in searching programs
update comment typo
change cwd to home before executing command
update sys plugin with new code
add pinyin cache for control panel plugin
revert changes to other plugin will refactor as a separate pr
websearch convert search word to url encoding before opening the url
prefill pinyin cache for program plugin
fix dynamic assembly
use shared command and open in new browser window
fix file not found ico png
revert use sleep instead of ping
merge pull request from kingwl master
close encoding issues with everything plugin
better mutex cleanup
when both icopath and icon delegdate are empty null it will raise an exception
fix logger info
catch error early
fix searchsource editing when action keyword is not changed
use default thread pool settings
support user defined theme files in appdata close
skip calculation of pinyin score if source string is too long
fix highlighting for uwp results
revert remove first chance exception
fix typo preventing last pr from taking effect
remove useless line
remove useless code
fix a issues that jsonprcaction could be null
fix leave open option for cmd plugin
better path handeling
merge pull request from dgoon master
force english exception
add more runtime info
websearch fix messagebox behaviour
no link target redirection
image loader is not thread safe
enhance cpu type checking
make image created in hash freeze
better error log
fix a null references issue that may crash wox
catch more json exception
fix exception language
remove dropshadow since it s buggy
fix to allow plugin to start up disabled as default
fix download plugin
update main cs
rename log file
providing user agent for http download method
revert accidental removal of interface methods
fix missing using directive
fix commit multiple selectionchanged firing
remove a duplicate check
exposes reload command via ipublicapi
fix index out of range exception
fix wrong comparison parameter
fix incorrect behavior with app paths
remove unnecessary decoration
check uwp names
remove context shell load codes and change image cache path
add global unhandled exception check
merge pull request from theclueless upstreamdev
fix crash issue when user clicked cancel in uac prompt window
remvoe useless code
add browser path to open method in browser bookmark plugin
fix typo in helloworldcsharp
add f1 to open wox function guide page
add trailing slash to saved folders action for folder plugin
fix non exe lnk
ensure updatemanager is disposed
add hide query window before actual reload show success msg
let it crash
add protection when creating uwp
fix everything not support keyword with space
use system web proxy
fix missing query part when using global wildcard sign
remove unused stop watcher
cater for debug mode
fix single instance
ctrl enter runs application as administrator
add missing using directives
dispose when restart
set hidewhendeactive default as true
debug should not send error
fix websearch setting issues
programs fix messagebox behaviour
change everything default search mode
focuses first textbox on load
re enabled exception during debug
fix win r issue
fix a crash issues when inputing after wox start
resultpanel should be empty when there is no result
improved firefox bookmark parsing query
add folder location display
fix theme font changes both whenever one is changed
fix system nullreferenceexception for
fix the order of everything results
add threading task run for startprocess
remove unused code
lower the score for folder results
check before computation
fix calling null callback hookedkeyboardcallback
add reload call for programs plugin
fix miss default font
change method name to make intention clearer add additional methods
merge pull request from theclueless uwperrorfix
add handling when query action keyword is empty
fix log logic
remove console target
ensure updatemanager is always disposed by wrapping it in a using block
fix disable plugin not working
remove unused new list
merge pull request from zhiyue dev
add more programs
fixes issue indexoutofrangeexception when reading empty icon from registry in controlpanel plugin
add ireloadable interface and method
fix bug from merge in folder plugin
move disabled plugins to the bottom of the list
updated user selected record
update upgrade feed url
add universal pretty print method
fix the origin root directory of web search plugin icon images directory
update setting description summary
merge pull request from theclueless smallfixes
revert try list appveyor installed apps
change https to http becuase the ssl has expired
fix crash on invalid chars in path
expose reload data call in publicapiinstance wider usage
remove duplicated code
fix uninitialized ui
display plugin icons in wpm list uninstall
make python plugin enabled by default
catch update exception
fix a typo suggeestion suggestion
add equality checking for old keyword and new keyword
remove useless exception formatter
fix controlpanel plugin search bug
remove extra leading space char
optimize for text selection after pressing win r
fix issues with aero
add method used to indicate deletion of data location
remove unnecessary unit test
update controlpanel plugin with new code
update removeactionkeyword method
update allsubstringscontainedincomparestring calculation as per comment
fix a bug when creating config directory
fix a crash issue
fix closure bug
add support clickonce application
auto change log level
more exception info
remove sensitive information
update comment only
add nonnull check
speed up websearch plugin when suggestions turned on
fix accident bug
merge pull request from zsxsoft master
fix improve the weight of user selected items
fix that directory could be remove by user when wox running
fix log format
uncommented and remove junk code
fix websearch duplicate result
set port number if not zero
merge pull request from griccardos websearchscore
remove useless code
save all plugins settings on settings menu close
revert limit calculator trigger must has one number at least
remove useless code
add more comments
use already load resource
fix crashes for context menu on empty query
bugfix for issue
remove useless code
add possible exception path too long handling
fix add button action to not duplicate implement new model
remove single white space no code change
merge remote tracking branch origin master into bao
fix prompt restart wox after uninstall plugin
update themes url link
when user querying thirdparty plugins system plugins shouldn t be included
move comment to pr
disable blur when os is not supported
fix shellexeccmdline fucntion using global system diagnostics process start to run program without working directory specifying
fix missing required parameter in logger
fix default icon path when add new web search
update class to inherite from settings programsource
per comment simplify code in setter
disable more logs
typo in log
add known error when trying to retrieve uwp program s thumbnail
more debug info
fix everything hight data index
add win32 debug info and change log to debug
fix build fail from task reindex run
ignore broken cache
bug fix isselected is not implemented in resultitemviewmodel
handling chrome browser not installed
issue if wox was set to ignore hotkeys on full screen it would fail
fix some bookmark plugin issues
fix duplication of tray icons
fix logger level
merge pull request from zlsun master
error report windows won t crashs
merge pull request from wox launcher dev
reduce load program size
save startwoxonsystemstartup setting
change icon for thirdpartyplugin indicator
catch borken control panel item
add tips string to folder subtitle
bug fix the caret is not at the last of search text when switching back from context menu
fix error message logging for prod run
fix can t delete items in folder plugin
remove unused code
temp fix for cmd hotkey
fix characters not being properly escaped in python plugins
allow plugins return null
allow pluginmanager class to manage the reload call
rename cache file
fix pageup pagedown for result list
merge pull request from wox launcher bao
update search wild card
fix usermodelid and famaily name
merge pull request from cxfksword master
more log for image load
fix a http request issues
change init order
simplify the check logic
add file length check before loading cache and remove binarystorage save when file was already locked
convert to in query strings
simplify condition as per comment
remove first chance exception
handle query exception
parallel foreach list is not thread safe
message to activate should only show when disabled not when enabled
change method name for cleaning up after portability update
add fix height for caculating windows top
fix null error when plugin doesn t set action
open url in new window browser
add issue link to comment
keep main login clean
fix uwp oneconnect
merge pull request from zachrammell unix paths
now checks if path has trailing slash before add a new one
load multiple times instead of one
allow comma in input expression
reduce lock scope for better performance
skip broken folder when re index
add settings saving to file
fix variable state which failed to represent correctly
program plugin ignore add for subtitle
remove test code
remove bing search suggestions
use sleep instead of ping
add various keywords to calculator
remove unused packages
fix a bug with windowsshellrun
add log into alphabet service since it can induce a large perf hit
fix python plugin crash issues
contron panel item priority should be lower when compared to programs
try fix version detection bug
check cache per hour
merge pull request from jhdxr master
update score when titles are same
fixup test code
temp hack for folder plugin results
remove unused code
enable auto update
make debug mode clearer during exception
fix performance issues while loading fonts
merge pull request from barakyaari master
merge runtime info
collect more plugin info
fix delete logic
remove useless code
add check null for event
forgot to remove some lines
faster startup time for control panel plugin
add sqrt as calculator keyword too
remove browserbookmark subtitle wording boomark
remove white space
more exception info
fix line ending
fix the bug in windowsshellrun
merge pull request from griccardos plugins wox plugin pluginindicator disabledfix
update test with updated method name
cache should not be same name
filter duplicate program item
add restart wox command into sys plugin
fix ui flickering issue
fix can t find file exception caused by abnormal scale
remove useless log
update query defualt to string empty instead of null and update result plugin id to be internal
uwp fix issue where if itemssouce is initially null listview would never render
add condition sdkint jellybeanmr1 for set flow direction
ios redraw frame on cornerradius changed
cgcolor creation fix on ios and macos
apply the increment to all ctors
make sure platformconfiguration created with stepper
handle geometry for master canvas to move up layout timing
ios export mdp tablet renderer fixes
fix profile frameend call
ios fix issue with webview dispose and resizing
ios make sure to include query path on generated url
tizen improve the switchcell performance
ignore layering test on android
try to generate more unique names
add null check when tearing down renderer
changed stepperpositionproperty to public
changed gtk animation class to be internals
android fix a potential null reference
uitest add preserve to issue uitest
make test button visible on uwp
android fix possible nre when detaching navigationpagerenderer
tizen supports switchcell oncolor
ios fix nre clearing listview contextactions fixes
keep tab image collapsed if no datasource is available fixes
make shelltablayoutappearancetracker public
fix button tests for fast renderers
xaml avoid ambiguousmatchexception
use correct reference frame for gesture origin recognition
fix map itemssource bp name
mac fix progressbar
c seal our attributes
ios issue warning to console when masterdetailpage is pushed onto navigationpage
apply fix from to style cs fixes
remove nonsensical comments
add preserve attribute fixes
macos boxview color fix
macos fixes issue with hiding master page of masterdetailpage fixes
uwp fix transparent default button
fix wrong namespace in fontfile class fixes
fix position of placeholder
remove dead code
automated dotnet format update
c throw meaningful exception on duplicate rd key
automated dotnet format update
update itemsviewlayout cs
fix profile frameend call
add null check to getrenderer
fix unbalanced performance start stop call
add updateapplyshadow in viewdidload fixes
tizen update listviewrenderer for header and footer
xamlc change module mvid
wait for test to finish so it doesn t break later tests
macos set default bezel style for button
export animationcontrollerforoperation to fix hot restart
ios scrollviewrenderer clear the renderer
tizen fix carouselview issue
add parameter attribute to constructor parameters
revert update forms cs
tizen fix custom font issue
gtk fix navigation
uwp fix issue where label formattedtext was crashing
u send log to trace
set android app to not clear data on restart
update carouselview visibleviews default value
don t hook up unloaded handler in uwp pagerenderer until loaded fires
execute the test on android
don t reset implicit style
android add missing unsubscribe on activity destroy
android use non deprecated constructor for gesturedetector
add default parameter value of zero to inavigationpagecontroller peek
automated dotnet format update
dispose tap gesture
previewer make sure we do not crash even if the previewer doesn t s
reuse handler when invoking on main thread
handling page title change for bottom navigation view
make listview cachingstrategy property visible to intellisense
temporarily ignore intermittently failing test
automated dotnet format update
controls fix build of test case
hr properly reload merged rds
enable ios test for automationid on toolbaritems
core allow registrar registerall to be run multiple times
winrt add base platform as inner proxy navigation to application fix back button issue
add isinlayout check
fix the tests
change accessibility modifiers for itemdecorations
tizen fix carouselpage focus issue
make toolbaritem constructor match properties
the dx for the shadow isn t being accounted for with the paddingleft of the background drawable
uwp make sure to update back button on tabbedpage appears
uwp controls are ignoring their rendering routines
css letter spacing
fix nullreferenceexception in motioneventhelper for android
uitests disable test on ios12 for now
automated dotnet format update
gtk fix layout calc with navigationbar fixes
ios use absoluteuri to parse tel uri containing spaces
uwp wpf fixes rendering of the main page fixes
ensure constructorcountid and timeid elements are available before querying them
make onplatform helpers public
ios check for selectedindex value in onended
x recover from memberaccessex
c fix css mapping
wpf fix formspanel measureoverride negative requested size
add ios version check for crashing code fixes
change image source to one we control
if the element s renderer has already been nulled or disposed request zero space for it
c fix bp declaringtype in textelement c
android null check previous page on tabbedpage
bugfix catching reflectiontypeloadexception in the visualtypeconverter
tests make sure the test pass even if the locale is wrongly implemented on the platform
fix random test failures
macos autolayout engine exception fix
android begininvokeonmainthread should not possibly block
fix a mediaelement crash trying to reproduce a video without internet connection fixes fixes
android java lang runtimeexception animators may only be run on lo
clear cached cultureinfo and timezoneinfo on app resume
undo broken change in un watching empty view observer
verify that the test is on screen before restarting
dispose check on buttonrenderer
ios throw better exception on missing bgimage fixes
ios fix imagerenderer dispose
uwp prevent crash when using datetime minvalue
uwp optimized zindexing for a thirty nine percent speed boot on loa
update colorextensions cs fixes
add masktobounds property to clip a layer s contents fixes
fix navigation to expander test
make sure stepper buttons are always the same size fixes
add instructions fixes
only post disabletimer to the queue if it s not on the ui thread
android restore and obsolete editoredittext and entryedittext controls
fix bug commonstate focused issue
ios update all subviews of the indicator when currentpage changes
add allowmultiple true to querypropertyattribute
fix up the merged file
ios fix crash when pushing page on stack on orientation change
uitests wait a little bit longer for go to test button to appear
android fix a nullreferenceexception that happens when a page is closedd fixes
winrt do not store previous page when popping
android set dialog to null when dismissed
fix reflection call uwp
fix crash when itemssource is set to null
add null check in updateflowdirection
fix incorrect bug description
remove unnecessary ok button check that won t pass on older android versions
gtk fix padding margin properties of frame element
update wpfplatformservices cs
make createnativecontrol virtual instead of abstract
macos boxview binding to backgroundcolor broken
a don t fail on android api obsoleted in alpha
unsubscribe onmodelchanged event when tableviewmodelrenderer is disposed
controls only register appcompact renderer if running in that context
only set elevation on lollipop and up
implemented openuriaction in wpf backend fixes
automated dotnet format update
gtk do not show a label in gtk slider renderer
attempting to fix possible race conditions in automation id tests
clear mainpage renderer if exists after reload
windows correct enter esc keypress behaviors for messagedialog
shell fail when trying to add a duplicated route
add missing preserve attributes
allow intellisense to discover matrix
fix swipe gesture not being handled in visualelementtracker fixes
tizen make activebezemelement as readonly
a fix typo fixes
stop the madness
ios fix transparent modal issue fixes
fix itemspacing numberofsideitems peekareainsets and snap in uwp carouselview
fixes issue when setting navigationpage sethasbackbutton after page has navigated android appcompat
add disposed check on fastrenderers framerenderer
fix device error message reference to xamarin forms init
xamlc fix message for unresolved event handler reference in datatemplate
gtk fix navigationbar check for layout calc
uwp use rawpixelsperviewpixel for scalingfactor
make testcase visitandcheckforitems public
tizen update style for phone idiom in imagecell renderer
fix swipeview crash with nullreferenceexception
remove matchparent value from property editor
ios provide a base carouselviewrenderer
add missing virtual keywords
correct minor typo
make apis virtual for extension
distributedcachestatedataformatter should handle failed unprotect workflows
minor updates after mtls review
update endpointrouter cs
slightly more precise error message
fix test user profile service throws on invalid sub
changed identityservername constant to identityserver4
update mtls disco element
fix incorrect log
set color theme
exclude flakey test
integration test fix to use the same session cookie name as the actual code is now using
fix validationkey kid for certs
make addinmemoryusers accept an ienumerable
add invalid target token request error
use sid as sub for windows authn
remove status code check from setting cache control header
cleanup account controller
add callback page for js client
name of the session cookie is derived from effeciveauthenticationscheme which makes it configurable
fix tests for local login
changed the comment to the corresponding default
remove unused route paths
missed one more obsolete call
add todo for ro handling
fully qualify constants
fix comment typo in httpcontextextensions cs
null checking in events
merge pull request from martijnboland x platform certpath fix
set same site mode to none for authentication and session cookie
add parameters to introspectionrequestvalidationresult
typo in log
fix invalid grant type validation result
uppdate log filter
allow client to use prompt none
add defensive check to fix bug for when session is expired
clientauthenticationfailure event constant duplicate
add rt settings
only emit session id cookie if check session endpoint is enabled
skip events tests
disable windows by default and add a note about iis requiring configuration
made granttype public
remove html and body tags from federated signout iframe rendering
remove unsafe inline from authorize response
encode client list in cookie
make resource extensions public
remove issuer uri as accepted aud for priv key jwt
tests to allow empty grant types collection
add properties dictionary to parsedsecret
use constant instead of string
add toarray calls in resources ctor to resolve collection
remove extra blank line
ensure consumed time is utc
inject dependencies into mw invoke params
fix some todos
fix multiple www authenticate to one
include properties collection when cloning apiresource
add frame ancestors to csp on authorize result for form post
add null check when unprotecting data
change error code in tokenvalidator class from invalidtoken to expiredtoken
add di validation to host
more token failure event cleanup
handle unknown idp at login
add hosting logging back
ensure session id cookie is kept in sync on every request
remove unused handler
typo in code comments
cleanup program cs
add null check before setting consumedtime
update startup cs
add toarray to not step on collections
append to any existing vary response header when setting response header
remove unused code
remove old comment
rename ws fed idp
only revoke specific refresh token not all for client
fix exception with prompt login
fix typo in the x frame options directive
update log message
add forwarded headers to enable nginx hosting
add defensive null check
for authorization interaction oirc error response don t log an error to log file
more defensive check for externalloginscheme
re enable skipped test
fix xml comment
null check for device flow interaction service
loginformation changed to logdebug
update comments on consent controller
add isactive to testuser
fix incorrect comment about default grant type
add missed log error
remove redundant secret description access
filter out scopes not requested
fix issue redacte client secret in events
explicitly set the default value of base target in html response from authorizeresult
update startup cs
update authz policy to new scope name
add shared scope
don t show logout prompt if user not authenticated
skipped failing tests for now
remove unused mw hooks
add addacrvalue helper
fix typo in consent options
add more defensive check on check session endpoint
make authorize response generator public
consent page add fallback to clientid if clientname is left empty
remove extra out of index msg template that cause exception during logging
made client api secret validators public
make event classes public
hack for social providers throwing on logout
add check to scope validator for missing identity and api scopes
updated serilog message template
moved jwks endpoint
harmonize result in drts
remove superfluous url encoding
add meta element to authz response page
bypass hrd if idp sent
add button for noscript on authorize response form
move logging before removal so the promptmode is included in the logging
all setting input length restrictions
fix google login
use getidentityserverbasepath instead of request pathbase
made endpoint public so it can be used in custom routing
bug in consent when user denies
fix build to work with spaces in the path
support setting consent response
fix userloginfailureevent raised with interactive true in resource owner grant flow
force data load
remove web listener code
remove explicit issuer
update unhandled exception logging
add obsolete to custom token validator
made message tmodel ctor public
do null check around whole code block
include identity resource properties in getallresourcesasync
update ef seeding sample
add a comma in log message
revert remove unneeded newlines
remove unused method
disable clean for now
fix sid for existing logins
persistedgrants missing index on expiration column
prepare for breaking change in jwt lib
found missing introspection test
small spelling fix
improve obsolete message
fix join logic bug which duplicates the api resources returned
change log level from warning to information
update js client for code flow
add builder method for custom token request validator
relax frame ancestors for authorize response
add token types to public constants
add condition to logging in introspectionresponsegenerator areexpectedscopespresent
merge pull request from identityserver features obfuscate sensitive data in logs
default remember me to false
remove usage of obsolete api from ms idm
fix return url pathing when redirecting to external login host
resolve merge conflict
correction to spelling in logmessage
use task completedtask to reduce allocations
remove unnecessary using statements
add helpers for registering parsers and validators
add missing doc comments
merge pull request from identityserver dom fix consumed time
update redirect uris for sample
configure cookie policy to omit samesite none when request is not https
improve logging when grant validation error occurs
add missing fact
use consent required for when prompt is none and consent is required
change oidc client to allow revocation
fix exception message in key material sercice
rename authn schema
more defensive use of subject id in test user ropf
fix xml comment
fix client scope access
change xml doc
add comment about same site none for external auth cookie
reduce log level for expired secrets
use obfuscate extension method
remove ws fed
change rt settings
remove redundant call to gettype
updated consentpageresult to use getidentityserverbasepath
defaultclaimsservice getidentitytokenclaimsasync uses wrong resource parameter for profiledata
only re issue session cookie when client add
remove superfluous call into web host builder
fix typo in tokenerrorresult cs
fix typo s
extension to register icustomauthorizerequestvalidator
set console title
create new collection when implying all scopes if none requested
make authorizeresponse public
disable automatic on external mws the best we can
round trip authorization params
use generic service registration
create jwk document when signing with jsonwebkey
set default error for token error model closes
add property for authenticated request
fix route value
add expired helper for datetime
corrected value for parsed secret type
add js client reg
add more detailed version logging
fix broken query in getalluserconsentsasync
set the windows auth scheme from the microsoft kestrel constants
pass loginurl logouturl from identityserveroptions to auth cookie
add check to userinfo endpoint to make sure sub claim is present
guard against bad validator implementations
add config for both in and out of proc iis configuration
typo in constant name
remove enabled check on in mem client store
add comments around end session processing
spa checks for its dependences on healthcheck
fix xunit warnings
remove swagger from payment api
renamed irepository to iaggregaterepository
remove unused namespaces
fix merge issue
fix unsaved file on vs
merge pull request from geertvl master
minor refactoring for the createorder web api method and the entity base class
fix issue ordering event handler registration
add a comment about the simulated payment
fix crash when attempting to filter on only one item
fix merge issue
add a period to end of validation error message
declare payordercommandmsg class di
fix bad namespace from commandhandler
remove unnecessary await
fix issue logging out crashes
merge pull request from jo ninja fix spelling
minor comment change
add allowaccesstokensviabrowser in identity server we got an login error from xamarin
remove duplicate string to byte conversion
add marketing resource to identity
error fix applied the cookie policy to lax mode
remove unneeded local variable
rename property picturefile to picturefilename in migration designer
remove unused dbset
changed logout callback url
fix method name in ondisconnectasync
bug deserialization of integrationevent
renamed a variable
merge pull request from libphipp removefromgroupasync
remove hardcoded database name from graceperiodmanagerservice
minor refactoring order for the ieventbus interface
update profileservice cs
merge pull request from plneto fix log message
minor comment typo changeed
add get all campaigns by user id url
cleaned up the registration of domaineventhandlers and fluentvalidation validators
add some comments about immutability commands
remove task delay test
cleaning up the address valueobject class
update startup cs
merge pull request from jo ninja dev
minor change in exponential backoff config for the retrypolicy
add orderingdomainexception when the order doesn t exist with id orderid
fix authorization in webhooks api
add marketing scope to xamarin client
temporary fix for older clients
initialitze the list in constructor
eliminated the graceperiod logger message when querying as it is not needed and is quite anoying
remove hard coded values from identity srv configuration
changed logout redirect url
minor refactoring in method from servicebus to eventbus
change to task delay
merge pull request from eneuman dev
display the type name in the log message
minor refactoring in logdecorator
clean up and style improving
add private setters so deserializing on integration event handler works as expected
fix type in httpglobalexceptionfilter
customer basket items setteable
add locationid to locations model
loginviewmodel now stores the correct access token as an application setting
add token when call location api
fix registers missing and routes
merge pull request from dotnet architecture ardalis typo patch
cleaning up and fix typos
remove unnecesary catch
minor description refactoring
fix namespace top level statement
cleaning up the address valueobject
unused namespace remove
remove unused usings from modeldtos
identity server config clients to avoid consent screen on login process
remove warning on usings in marketing api
uploading the early draft xamarin patterns ebook
fix tests of basket
cleaned up buyerrepository cs
modify locationid in marketing campaing seed
change readallbytes to readallbytesasync in piccontroller
merge pull request from mvelosop fix failling integration tests
update basketcontroller cs
remove unused eventbusconnection property
fix ordering tests
increased number of retries
fix failling unit tests
replaced endpoint with copy
not needed since useidentityserver adds the authentication middleware
setting iwebhostbuilder and iwebhost as explicit instead of var in buildwebhost from the generic ocelotapigw
clarified endpoint uri
fixes order checkout
fix the missing sub claim
fix bug about loggerfactory and configureauthservice at marketing api startup cs
fix to close issue
fix spelling commited committed
fix for wrong items per page calculation for the view
solves error on order detail related to
fix mobile tests
replace loggerfactory addconsole with loggerbuilder addconsole
remove string format s
remove unused build service provider
remove machinename from logger configuration not really useful for eshoponcontainers
fix two typos to the filter
remove private setters to make class immutable
remove private property setters in address value object to make it truly immutable
remove unused method
converted tabs to spaces
fix unit test
fix namespace copied pasted from the locations api service into the marketing service
update mediatormodule cs
merge pull request from neilq fix eventbus test classname
a tiny fix of the issue
fix error merge
merge pull request from tthtun dev
private readonly string changed to private const string
fix deserialize error
remove not needed code
remove async warning
do not throw when job queue contains jobs in non enqueued state
fix test method to use the same time
checking for arguments key existence in redisconnection
make core process classes internal
don t use formattedlogvalues class in aspnetcorelog
fix urlhelper that s working in owin environment
fix job history sorting
escape quotes in raw json arguments
update sqlservertimeoutjobfacts cs
prevent expirationmanager s sql command from timing out
don t leak connections when distributed lock can t be acquired
fix null value is not shown in job arguments in dashboard ui
small fix for hangfire dashboard pages namespace description
fix cannot release the application lock exception in sql server
remove unit test for cron getdescription
setting servercomponentrunner thread name
don t decrement total succeeded jobs metric as it s unexpected
fix daylight saving problem by using stopwatch
create default options in sqlserverstorage sqlconnection ctor
add missing using statement
don t display max retry attempts count for processes it s always the same
minor comment change
each expirationmanager pass uses distributed lock to prevent deadlocks
stack trace parameters css class typo
comment for ibackgroundjobclient storage property
re order routes for enqueued jobs fixes
remove unused ctor from jobfilterinfo
fix queues page
bringing back expirationmanager to sql server storage
sqlstorage now posts server startup time
don t acquire page lock during counter aggregation
make state handlers internal
remove unused stuff
fix error in the dashboard when rendering an array that contains a null element
preserve stack trace when rethrowing operationcanceled exception
method name fix
use rowlock table hint in job fetch query to prevent escalation
fix unit tests
fix serializationhelperfacts by using the correct method
remove russian language html setting
authorize dashboard requests if no remote ip address found in core
fix sql server test
add unit tests for explicit interface implementation in fromexpression
make some dashboard classes internal
fix statecontextmock class
speed up msmq fetching after fix priorities
public access to the configuration
increase error log level in automatic retry wrapper
remove services that are inconsistent with server options
converting args to invariant string
don t calculate pollinterval on each iteration when using transactional fetch
update isolation level for integration tests
hangfire assembly is not cls compliant
add forceseek table hint to a dashboard query
fix cron cs for recurring jobs using intervals
specify parameter names for expirationmanager s query
cache client in the backgroundjob static class
perform class is obsolete now
hide some internal razorpage members from docs
add timespan zero timeout for suppress scope transaction
attributeusage for filters
add specific reason when zero retry attempts were set
remove recurringattribute references
set read committed isolation level explicitly for stats
checking queues array length in backgroundjobserver fixes
update countersaggregator to use clustered key for temp table
remove createprocess method from istatemachinefactory interface
add nolock hints for sql server job queues monitoring api
update invocationdata cs
changed id to id in sql scripts
remove unnecessary changes
signal newiteminqueueevent only for sqlserverjobqueue
fix build error due to wrong namespace included
add missing transaction to keepalivequery
show generic method arguments in dashboard
update layoutpage cshtml cs
add conversions between appfunc and owinmiddleware
make sqlserverdistributedlockexception class public
fix nullreferenceexception when owincontext auth user is null
fix servercomponentrunneroptions validation
changed id to id in sql scripts
prefer to use the defaultretryattempts const in globaljobsfilter
fix flapping unit tests
revert authorization filter implementation
set ceiling value for workercount closes
jobschedulepoller is now internal
restore tests that were testing obsolete members
simple fix for
validate the queue name in the dequeuedjobswatcher relates to
don t use clustered index on table variable in countersaggregator
prevent worker from processing a job on shutdown
return inner component name in retry component wrapper
checking for an old job format in the getqueue method fixes
statisticshistoryfilterattribute add by default fixes
simplify worker class instantiation
updated userabbitmq extension method to actually use the rabbitmqconnectionconfiguration virtualhost property and to correctly pass it into the connectionfactory
wait for job data during state changes fixes
add missing assertion
abort server component thread on join timeout
update serverspage cshtml cs
improve state reason in delayedjobscheduler for enqueued jobs
make awaitingstate handler class internal
disable immediate execution of the keep alive query when processing a job
remove confusing separator between type and method
return early from recurring job scheduler when shutdown is in progress
change infinite loop to not to throw exceptions
fix the consolesample project
remove unnecessary line in jobcreationprocess
update taskextensionsfacts cs
lazy redisclient connection
return a message instead of throwing an error in htmlhelper displayjob
update corebackgroundjobfactoryfacts cs
return non highlighted stack trace in case of timeout
fixes automaticretryattribute issue thanks to unit test
fix dashboard routes for css and js resources closes
make cachedexpressioncompiler class internal
make htmlhelper formatproperties method obsolete
remove recurrentjobid from items
add verification to job expiration check interval
remove unnecessary colon in argument renderer
make sqlserverobjectsinstaller public
enable remote request authorization for samples
show inner exception instead of aggregateexception when performing a task
don t use new c features as they aren t supported in appveyor
add readpast hint for expiration manager queries
shutting down manager thread only when job has not been fetched
interesting todo for the job class
add job tostring method back
add some notes to the istate reason property
permit dashes in schemaname
add some hangfire sqlserver options for the sample project
merge pull request from seek jobs feature no tablock
add test for additional timeout validation
fix post method handling in commanddispatcher
stoptimeout now defaults to zero seconds for backward compatibility
deprecate the jobactivatorscope innerscope property
add a todo note regarding addrangetoset to rewrite it using merge statement
merge pull request from fire eagle version fix
set newiteminqueueevent only when addtoqueue called
make queue name accessible from recurringjobdto
remove failed job logging from the worker class
add some seealso tags for enqueuedstate
make all the assemblies cls compliant nothing prevents to do that
make backgroundjob clientfactory internal
remove unnecessary check from job class
fix sqlserverjobqueue s getenqueuedandfetchedcount method
disable flapping test
make getrangefromset to be compatible with the newer schema
fix redisconnection getstatedata methods fixes
add coverity scan suppression comment
remove the nameof operator
test format of datetime arguments
fix queues page in sql server
ensure recurringjobschedulerfacts don t depend on compatibility level
remove dispose method of the redisjobfetched class
change request expiry date
remove unnecessary line from preservecultureattribute
add jobstorageconnection addrangetoset
copy job parameters in createcontext copy ctor fixes
set default server component timeout to infinite
free dispatchers collection type is indifferent
remove the unnecessary key output in trimlist query
redis integration tests fix
fix xmldoc tag for succeededstate serializedata
make sqlserverdistributedlock class public
fix obsolete useserializationsettings comment
some suspicious places demistified
make jobhistoryrenderer class public
fix failed unit tests
remove unused methods in messagequeueextensions
fix flapping scheduledstatefacts unit test
throw an exception telling the user to make sure to first initialize the storage first
pass dashboard context to jobdisplaynameattribute format method
increase distributed lock timeout for continuations
decorate all the exceptions with the serializableattribute
add correct type parameter when finding serilog s isenabled method
fix awaitingstate dashboard renderer to display options as string
add interface usage target for the jobfilterattribute class
another test fix
remove unnecessary join in sqlserverjobqueuemonitoringapi
use aggregatedcounter table only for historical statistics
bring sqlserverobjectsinstaller requiredschemaversion back to avoid breaking changes
add getinstallscript method
fix reason for failed state transition in delayedjobscheduler
remove redundant counter
remove obsolete methods in the jobmanager
move storage s removeset method to transaction
remove readonly keyword from the jobmethodcallrenderer maxargumenttorendersize field
increasing time out interval for fetcherfacts
state reason argumentnullexception fix
test method rename relates to
create enum property and test for retry attempt
make static typeextensions class internal
deprecate the iserverprocess interface
fix for redis monitoring api
revert unnecessary changes in program cs
servers records don t expire now
remove nameof operator
add possibility to pass performcontext to background job method
add the blank line that was remove by an accident
fix for catching distributedlocktimeoutexception based on resource name
dashboard middleware url fix
allow passing null state to createcontext
make taskscheduler default the default scheduler to avoid breaking changes
remove machinename validation fixes
the job is unable now to throw unhandled exception from the dispose method
shutdown timeout fix
remove jobs from processing queue starting from the head
fix wrong pagination on scheduled jobs page
remove unused using statements in a solution
don t cache backgroundjobclient instance in a static field of the backgroundjob class
fix counters deadlock
don t render arguments whose size is too big in dashboard
remove unneeded commented sqlparameter
throw original json exception when typeconverter failed to deserialize argument too
automatically resolve dashboardoptions timezoneresolver when using net core
add experimental support for sql azure
createworker method is now virtual
fix error message typo missing t
add synchronization for transaction method calls in sqlserverfetchedjob
implement iserializationbinder for newer versions of newtonsoft json
made comment clearer
validate the queue name in the schedulepoller relates to
merge pull request from yngndrw argumentrenderer timespan datetime
small fix for sqlserverobjectsinstaller
lowercasing path info
make new jobstorageconnection methods virtual
make into public prop
use default or parent transaction isolation level
add diagnostic message for flapping unit test
revert remove jobs from processing queue starting from the head
merge pull request from hangfireio retry install deadlocks
allow to use zero based poll interval when sliding invisibility timeout is used
set content type for razor pages fixes
remove length limitations on sqlserverdistributedlock timeout
fix the order of fetching for prioritized queues in msmq
add transactiontimeout option for sqlserverstorageoptions
update cancellationtokenextentions cs
put obsoleteattribute on obsolete public types
allow idempotentcompletion to move to non final and non processing states
place obsoleteattribute on servername in server options
change default language to english
add support for dashes in queue names
add missing nolock hint when determining the queues
change exception message
revert don t decrement total succeeded jobs metric as it s unexpected
don t log error in case of an exception in worker itself
persisting the job after unapplying succeeded state
check continuation age before throwing timeout exception
remove unused using directive
remove accidental virtual keyword
try exact format first when deserializing datetime values
race condition fix try
remove istorageconnection deletejobfromqueue method
change visibility setting for use method
allow to use patterns for queueattribute based on job arguments
remove the with tablock hint from the expiration cleanup
fix globalconfigurationextensionfacts to clean up serializer settings
fix the expirejob_setsjobexpirationdata unit test
remove unnecessary test
add a copyright notice related to stack trace highlighting
fix potential deadlocks cause by suboptimal queries in sqlservertimeoutjob
use serilog sourcecontext naming convention
fix elmahlogprovider to show errors when using sql log
fix retry attempts count
fix broken unit test
make perform context constructors public
remove unnecessary connection factory instantiation
unit test related fix
move nocount on statement before the delete in countersaggregator
set response status code on authorization failure
add readpast hint to the job fetch query
remove unnecessary attribute usages
fix xmldoc to point to an actual parameter type
deprecate the iservercomponent interface
add todo for the backgroundjob class
merge pull request from devmondo master
aggregate counters only if there are a lot of them
fix unit test that used string instead of type
add userecommendedserializersettings method
use hostname environment variable for unix
fix tests on mono
fix nullreferenceexception when rendering null ienumerables
fix execute_startsalltheprocesses_inloop_andwaitsforthem test
return storage instance from usemsmqqueues method for chaining
fix msmq test to use message label instead of body
fixe code identation
distributed locks in sql server storage respect schema name
merge pull request from gaiazov master
make creatingcontext ctor public
be prepared for server ondispose
decorate all the remaining exceptions with serializableattribute
remove assemblyfileversion attribute so product version and file version on the dll match the assembly version
make filter attributes sealed
don t leak colors in colouredconsolelogger
use read committed transaction isolation level
fix bug in history when duration int maxvalue
revert wrap job creation process with distributed lock
update layoutpage generated cs changed in previous commit
don t include the statemachinefactory class to the documentation
decorate enums with type name in dashboard
job filters can be applied to methods
simplify writing of embedded resource to response body
minor documentation changes
make dashboard htmlhelper class public
trim trailing slash in dashboard path if exists
move throttler delay call outside of connection and lock
add test for invocationdata to check all code paths
remove unnecessary query condition in sqlserverjobqueue dequeue
log an exception before try to re queue a job
convert lock timeout value from double to int
undo change internal
updated general settings model
add null check in selectprevtabitem
allowing launcher to be run as startup project with no command line args
fix nit comments
powerlauncher fix getting theme changed event
working app execution
fix saving to json in pt settings
updating packagerepository comment based on pr feedback
user ryanbod fix unittests
text is selected whenever launcher gets hidden and is then set to visible
codeflow cleanup when selecteditem is not null
fix bug in multi monitor scenario
catch security exception and access denied exception while trying to read a file
to catch uri format exception in pt run
fixes an issue where pointerexit would hide the selected context buttons
cleanup null reference fix with correct code practice
remove unecessary implementation of irepository interface as this is implemented by the base class
pt run string with accented characters search
fix some recent compiler warnings
add resource only when it is not already present
remove unused mutex string
add case insensitive hash code calculation
remove unecessary event name from settingsenabledmoduleevent
add in ignore for alt space
updating wox infrastructure to save to appdata local microsoft powertoys run directory
fix potentially infinite loop with logs in color picker
workaround for settings viewmodel tests
handle exception while editor is parsing custom layouts
fix unit tests
remove unused namespace
add support to run apps in development mode
updated query delay in launcher
settings add recursive delete for ioprovider deletedirectory to prevent crash on non empty directory
allow embedded images in svgs to be displayed
add environment exit statement
making open directory subtitle fix and the title as the current directory
improve readability of editorwindow
fix resizer thumbs not being shown
skip erroricon and defaulticon while resizing the dictionary
modify scoring algorithm for fuzzy search
defensive check for querytext being empty
searches for keyword only in title and not content to improve the quality of results
fix message format for exception
add missing to condition check
fix visibility property changed invoked on clicking away from pt run
fix sync issue in selecteditem
prevent clicking through when selecting a color using color picker
catch unhandled filenotfoundexception exception
sanity check for imported settings that may have invalid data
fix launcher not exiting when pt is terminated
fix for visual glitch
make transforming struct field names to dash case region invariant
fix powertoys run not reacting to change in settings
fix deleting custom layouts when closing on saveapply
fix wrong name logged in program plugin
use default options for utf8jsonwriter
fix release build
reduced the description weight by half
commit to re trigger github pr
pt run folder path starting with backslash
re add fix sequence empty exception on calling max in program plugin
add back in key for fz json
making folder results from the search indexer plugin behave like folder results from folder plugin
fixes null reference exception
initializing query text strings
null left rightwindowcommands
fix for result on empty query
remove python bindings
simulating win backspace key press instead of win control as it was launching ww
the open current folder result shouldn t change the query text to open folder path when selecting the results
apply correct theme at startup
comparing exepected vs actual json strings this fixes two issues
severity code description project file line suppression state
fix a crash in settings imageresizer
fix settings upgradeconfiguration code path
add in frontmatter and latex
folder results open the explorer window when selected
add null check for windowsxamlhost_childchanged handler
fix issue with autocomplete text remaining on clearing query
fix for null reference exception
last stylecop issue
remove readonly collection map
fix configuration exception message to make it clear what the first type in the message represents
parameters were not stored in querydatasourceinjection
remove unused method
fix duplicate assertion
skip generic mappings
works without as
pass for now
remove unused line
collapsing more expressions
remove console writelines
expression to var
it is tdestination already
trying with new mapper
update queryableextensions cs
workaround the error avoid inlining for value types
remove map method
just skip those that are already set
don t cache useless data
merge pull request from mikecole develop
harrcode the year
exclude the am assemblies
prevent useless casts
merge pull request from lbargaoanu docs
not used anymore
make debugging easier
remove expression mapper map
use concrete types
a bit better naming
fix typo in method documentation
remove unused ctor
unit test for mapfrom string
merge pull request from lbargaoanu efunmappedmembers
oops missed an extra cast
global ignores actually ignore now
a typo and some unnecessary work
add the configuration option to disable constructor validation via automapattribute
assertconfigurationisvalid profilename should throw if the profile name does not exist
merge pull request from lbargaoanu formember
remove hashset mapper map
just in case it s not a memberexpression
fix unused event warnings
you cannot include a type map into itself
remove primitive array map
generate it only when needed
no idea why i had to commit twice
memberstoexpand is optional
add clarification on usage
stupid merge conflict
remove collection mapper map
assume single automap attribute
add sourcemember to imembermap
i should run the tests before opening a pr
reverting debugging stuff
obsolete profile createmissingtypemaps
not used anymore
more code docs
missed expression binding failing test
confirming fix for when destination member is called type closes
refactor to use an config addfeature method to add the features
remove exception swallowing from property name resolver
add explicit validation test
add sorting of interfaces fix
be more precise about it
don t allow ambigous matches
remove assignable mapper map
merge pull request from milanio neutralresourceslanguage
i think that cast is made elsewhere
remove pointless compile
maxdepth doesn t set preservereferences
implemented open generic closed generic
only for the full framework
don t change anything just throw
pulling up some methods in to the interface
fix for the private setters in base classes issue
resolveusing one that was missing fix
if a destination was passed map to destination subobjects by default no usedestinationvalue necessary
that s not used anywhere
no need to check for null
expandoobject is a dictionary and should be mapped as such
it failed when the tests were not run in parallel
test was not complete
respect pathmappings in unmapped property names
it s never null
use the isgenerictype extension method
revert getdeclaredmethod order by parameters count
remove duplicate call to reassignvalue
remove possibility of recursion on reflection items for
merge pull request from lbargaoanu mappershouldreturnthesameinstance
forget about it
searching based off of requested not runtime types
restoring generic maps
use the seal method on the featurecollection
fixes issue without failing and current unit tests
add global ignores to all type map creation paths
core automapper now can stand alone
make private explicit
use our equals not the object one
add security attribute for some weird edge case
extra space for when n doesn t show
test with null source
expressionmapper now throws an exception if it cannot determine the source type
define comvisible attribute
reverting pulling up some methods in to the interface
merge pull request from adamralph fix whitespace
it s called newguid
add docs about how the include all derived maps works
one mapper instance per unit test
making the test pass
remove extra nulls
update queryableextensions cs
merge pull request from lbargaoanu dynamicmapsvalidationmessage
remove unnecessary using statement
add method which handles the body method extraction for both methods
private should do it
merge pull request from lbargaoanu extrathreading
it doesn t have to be dynamicobject
for each derivied type of an include the type should be add to included base types
corrected value resolver exception message closes
filtering dynamic assemblies
cleaner generated code
source member config clean
i guess we want only inline maps
set the destination member
skip it if we reach maxdepth
changing to use the generic version of iset
add case sensitivity option for defaultname comparer
measure only the map calls
gethashcode returns a precalculated value in an unchecked context have put the calculation back into a checked context
switching from struct to class
break long line in automapperconfigurationexception
preserve source type
add ignore to non generic imemberconfigurationexpression
skip validation when containsgenericparameters is true
check the reference not just the id
corrected convention matching
instance cache fix
clarifying docs around null behavior
converting value resolver type
remove an unused property for this test
slight change to wording
only the object overloads are affected
the generic version fails on net core but it kind of makes sense the generics are not base types
missed some expressions
update queryableextensions cs
remove dead links
conditions moved to single expression
add a link to the wiki
fix unit tests by moving setup to setup
remove concurrency from profiles no point
fix typo in the comment
code refactoring ignore reverse property the excellent way
fixes case where destination property is an ilist i believe getinterface ilist true returns null for the interface itself
fix bad merge
fix members with no getter
merge pull request from lbargaoanu master
add name identifiers to better understanding
it doesn t have to be a class
remove failing test from codebases that don t support it
it could already be true set by the attribute
fix unit test
include the property map when checking for equality
cleaner generated code
marking assembly as cls compliant closes
fix failing unit test
seal still throws
inline set value
reversemap should set memberlist none
one more test passing
static method without parameters
with new mapper and resolvetypemap instead of findtypemapfor
clarified documentation on difference between mapfrom and resolveusing
use a constant value to avoid a memory leak
we need a list
non generic version of usedestinationvalue closes
don t fail when the parameter name is null
remove redundant unit test
issue reopened for enumerable empty revised
check member types
collapsing some variables
just look for a memberexpression
char doesn t convert much too
only doing collections
confirming new config applies global ignores to profiles closes
use profiles property instead of _typemapregistry
on net core length is not a property of string for example a bug i guess
excluding tests for silverlight
fix issue with inotifypropertychanged proxy that the event target was the wrong object
merge pull request from lbargaoanu notneededcode
revert use the validating base class
obsolete mapper initialize
just consider the static types
mapfrom should throw
merge pull request from lbargaoanu typeconvertermapper
improving api comparison tool
add back trim behavior in replaysubject
fix test to ensure enumerable executes
marking some classes as partial
fix two warnings
task run already starts the task calling start throws an error
exclude coverage from corefx type
remove unused field
add a few more operators to the blacklist
restored comment in isdisposed get
mute these tests as they are responsible for deadlocking the xunit runner currently
fix short circuit to ensure we throw
reduce loop body in singleordefault
fields can be made protected
skipwhile don t init to default value
link cancellation tokens
marking lock field in switch as readonly
stylistic change to scan
add comment to sequenceequal
whitelisting conversions in api compare
sink dispose should only call dispose true once
update asynctests bugs cs
cleaning up some usings
fix spelling error in documentation comment
fix infinite recursive generics in catchscheduler
linking cancellation tokens
fix issue in groupby
merge pull request from danielcweber fixnamingoflocalvariable
tolookup release lookup upon failure termination
merge pull request from schmich fix spelling
singleasync don t init to default values
have proper cleanup of the enumerator
clean up assemblyinfo cs files
use async iterators in ignoreelements
avoid infinite generics recursion in synchronizationcontextscheduler
add some comments to timer
sealing a few more types
rename field minimize diff
use the count property on a list instead of enumerating it with count
ensure enumerator is disposed
merge pull request from dotnet cleanassemblyinfo
add asyncdisposable nop
rewrote this test to indicate that errors are on the observer not rethrown
include the cancellationtoken in never and amb disposing
reverse order of locks taken in localscheduler systemclockchanged possible fix for issue
some code consistency in buffer
fix error handling in buffer
improve thenby calls
marking a field in toarray as readonly
add overflow check to range
remove taskext usage in return
skip another test
grant cancellation in reverse
fix more missed out conflict resolutions
some code cosmetics
fix spacing in groupby
use static lambdas
remove more garbage
fix throw factory
use async iterator in generate
add missing null check to startwith
skipuntil remove unused helper class
fix a few typos
merge pull request from jamesdoran buffertimerraceconditionfix
sealing some more sinks
made ix async factory methods public
add some comment to select
fix sequenceequal chaining
todictionary clear the dictionary reference upon termination
push variable declaration into inner scope
fix build break in never tests
fix how assert is called so we don t crash the dispatcher
add comment to selectmany
remove a couple of old ifdefs
a task continuation should be cancelled when the sink is disposed to avoid leaking the continuation on long lived tasks
don t nullcheck getenumerator
elementat don t create exception after finding the item
merge pull request from akarnokd ixcodecleanup
use configureawait false when awaiting unionwithasync
test to catch onlyifcheap true
don t skip this
some code cosmetics
remove unused variable
merge pull request from danielcweber fixtaskobservableextensionssubscribe
remove dead code
add test for support of large enumerables
eta reduction of some lambdas
fix a test
fix asyncenumerablequery to support valuetask t aggregates
add review comment for orderby
use background thread for long running threadpoolscheduler
use stablecompositedisposable create for two disposables
remove unused var declarations
merge pull request from dixin master
merge pull request from danielcweber makesinkdisposableidempotent
fix invalidop when enumerating the systemclockchanged hashset
remove unused field
fix whitespace grmbl
update apiapprovaltests cs
remove unused allocation assignment
remove more dead code
sealing some classes
remove unused field
fix oom in partition code
one more approvals fix
simplifying code in synchronize
add another entry to the whitelist
add securitycritical attribute to iyielder unsafeoncompleted
save an allocation in threadpoolscheduler periodictimer
postponing some operator families
add review comments to sequenceequal
remove unused field
add test for double timeout exercises the cancel paths
remove the use of a createenumerator overload
remove unused using
window don t init to default value
a few more fixes
add missing using statement to unit test
null out _sourcects
clean up tests for throw
add a todo to playground
add diagnostic attribute
add missing state xml comment
fix build error
merge pull request from tcberryoxinst develop
optimizing layout of synchronize
remove unused methods
fix the name of a local variable
optimizing list creation in zip
fix argumentexception message
remove assertex throws
fix asyncqueryable zip
simplifying some logic in first ordefault
add constructor overloads to compositeasyncdisposable
scan don t init to default values
mark interface as internal
also put back volatile comment
improving takeuntil cancellation
remove unused using
use pattern matching
oftype use pattern variable instead of double cast
add missing header
add a todo
fix delay selector npe on immediate emissions
fix recursion bug
exclude test with await in platforms that don t have it
marking lock field in synchronize as readonly
improving cancellation of range
fix loop in first
merge pull request from blewzman asyncenumerabletakedisposeearly
no need for continue either
fix buffer race condition where timers could be created out of order and cause timers to stop firing
add missing await
merge pull request from reactive extensions etareduction
fixes throttling by passing through duetime
add missing configureawait
undid accidental changes to unrelated files
in enumerableex foreach index increment should be checked
avoiding use of fancy subscribeasync overloads
some code modernization
improving compat movenext async extension method to honor cancellation
add missing edit
remove imperative operators from whitelist
add a comment to multicast
fix the build
replace a static nop action that s unique per generic type parameter by stubs nop
use identitysink run where appropriate
add a comment to collect
trait instead of skip
using shared instance for the never task
count don t init to default zero
add missing copyright header
implement isempty directly
add missing copyright header
merge pull request from reactive extensions fix remoting test
clean up namespace
add missing copyright header
hardening an ix test where cancellation can bubble up
merge pull request from reactive extensions notaskex
modernizing some code
modernizing some code
sealing some more sinks
add remark to zip
code cosmetics in replaysubject
add covariance for iorderedasyncqueryable t
save one static per generic class field
asyncenumerable toobservable use try methods on taskcompletionsources to avoid unexpected exceptions when the iterator is cancelled
add a few comments
add comment to aggregate
some code cosmetics in multicast
use comparer for dictionary
remove taskext usage in never
fix merge memory leak
a better error message
remove unused using
takelastbuffer use base run explicit args
remove some obsolete code
excluding one more operator
skip hanging test
clean up test
minor tweak to toenumerable test
refcount don t init to the default values
check for existing type
closes compositedisposable fails construction when values contains nulls
some code cosmetics
fix a few typos
fix some test code
merge pull request from danielcweber possiblyfixlatest
add missing copyright header
whitelisting tohashset in api compare
chaining interval methods
clear the enumerator in onnext
mark a few methods as static
catch obj disposed exception
make sure caret is visible edit value control
change isvisible to isenabled
don t restore state if snapshot is the same
use a shorter type name without generic args
add correct using ns
set new version
support digit separators hexposition
add null check
asmed hex remove extra paren
check module too
don t create an object id if value is null
fix keyboard shortcut
merge pull request from silenthammerhun master
don t loop forever if bad metadata
update other activate window code too
update debug fail msg
pick closest char
create enum mirror if it s an enum
cancel inc search whenever a non supported command is executed
update thread pool check
update xml doc
don t send msg if process is terminated
dispose of location if it doesn t match the serialized location
don t show il code by default
update enum name
update title bar version
update wpf focus code
show the selected node
only include docs with modules
options dlg highlight all matches not just the first match
add a try catch block
update well known type code
don t reset zoom
don t raise object id changed event for hidden object ids
fix return value
show dll files if it s net core
ctrl c w no selection should copy the visual line not physical line
use another frame if exception frame is null
don t loop forever when type with bad metadata is right clicked
remove extra method overrides import loops
use visual basic instead of vb in lang combobox
add debug check for internal decompiler types in documentvieweroutput
fix comparison fixes
update vb value formatter
update failed asm cache
editor ops selectandmovecaret call translateto in case buffer got updated
if pos is between two refs prefer the right ref
compare decl types
don t set prop
hack fix for scroll bar context menu
use statement with smallest il offset when finding a statement
use equals instead of ref compare
type formatter always show namespace when formatting generic args
make props abstract
dispose of metadata
find the correct icomponentconnector iface fixes
don t clear selection if replace all didn t replace anything
filter out new lines if option is enabled
get gen param attrs from the correct type
add null check
return correct color verbatimstring
search ui make sure alt r a don t work in ui unless replace ui is shown
update asm simple name detection remove ni dll and not just dll
create enum info outside the lock
update uri tagger helper
remove assert and use runtime check
use digit separators
add a fix so the highlighted line is shown correctly if it was in a hidden tab item
compare content types
fix return value
check for field initializers
ctrl home end call caret ensurevisible
cache tostring state in type
make sure var homes array isn t empty
resize lang combo box visual basic used almost all previous space
check for empty ec asms
create a void if it s a fnptr
make sure this is treated as a parameter
add null check
add null check fixes
don t auto indent the caret when clicking somewhere before the auto indented location
refresh columns when invalidating a node with children
verify that it s manifest resource offset column
cancel incremental search if user presses enter
make uiutilities public
add null check
update kickoff method detection fixes
routedeventrecord read and write data in same order when serializing
esc doesn t close search window shift esc can be used instead
update color of rangevariables use parameter color
add missing enum value
double float tostring use round trip format specifier
fix overlap check
merge pull request from inlineasm
module breakpoints now derive from dbgobject
frame bps are temporary by default
save rsrc dlg box init filter property
use empty string if null
close file handles to the debugged file
don t assert in dtor if we re exiting
update order constant
highlight other refs even if current reference is hidden
remove measure override code
sighlp use normal content
ignore remove bps
mouse selection don t select a hidden line to prevent scrolling
call caret ensurevisible when setting new documentviewer content
add missing newline to xaml output
set new version
ignore invalid paths fixes
update dnf version
make sure in memory module isn t part of an assembly fixes
force projecttypeguids to be uppercase when written to csproj
increase search result limit
translate span to current snapshot
use full memory span if virtualqueryex failed
update disasm tooltip to include cpuid info
invalidate location column when language changes
make sure the red error border is shown in the ui when the ui is shown again
include previous spans
make sure nested files have been created
add debug fail calls if it s an unknown text editor command
write in if it s a readonly arg
fix not using bold italics from theme
move is already paused check
fix in param detection in c formatter
edit value clear selection before moving caret
force invalid offset if it s a native transition frame
include last windows error code in the exception message
update when changed bp code to support null values
update compiler error links
don t set prop if it s false
fix missing x key attribute
add esc clear marked reference command binding
increase the max count of recent files
show framework name in about message
use a local don t read the property
add more icons to context menu
add dbgprocess pointersize
remove extra spaces and semicolons
init debugging field
compare dir name
allow more than one instance by default
ignore invalid paths
make sure async methods don t exec on ui thread
completion make sure width of control doesn t decrease when pressing pageup down
don t use a string ref to the assembly
use path to native dnspy exe in registry explorer integration feature
use decl type of overridden method
show latest vs version
use module filename not name
prevent showing dlg box if it s already being shown
fix exception name so it s never an instantiated generic exception type name
add method spec
init line field earlier
unregister not needed in this case but for consistency
update title strings
fix order of modifiers
check for null pkt
add extra check
ignore stop debugging if engine hasn t connected yet
dnspy console force o path to be the full path
use inner most file
remove dead code
include host filename in command line
remove dead delegate
cache value to prevent object disposed exception
unbox value type
impl get enum names
update default roles of code textview
allow debugging more programs from the debug menu when we re already debugging
change default value
use utcnow instead of utc
add a ref to operator keyword vb so tooltips work
call gc after edit code dlg box closes
use ui thread
enable show pdb info il option by default
protect against disconnection exceptions
output window allow null strings
raise processpaused even if process didn t change
make class internal
use one lock
conding style fix
change default value
update apphost patcher
don t show search command in every ctx menu
don t resume if we re func evaluating
don t throw if type doesn t exist
move file check
update tfm strings
ignore version when looking up assemblies fixes
update vs version comment
update attach to process link
don t highlight change if it s an error value
use first text extent
make sure a ui ctor gets initialized on the ui thread fixes
remove ctx menu item command
update code that deserializes xaml numbers
ignore mda option if net core
minimize propertychanged notifications
close engine after runtime
merge pull request from smourier master
disable mono support fixes
repl clear selection when esc is pressed
initialize settings from global settings in ctor
merge pull request from slowlogicboy isvisibletoisenabled
make method public
sort dropped files before add them to assembly explorer
left sel margin don t scroll down if last line is fully visible
select full lines if selection started from offset column
add sort assemblies command to assembly explorer context menu
fix return value
use desktop asm comparer needed by unity asms fixes
remove useless check
init class field early since it s accessed by other thread
search don t move caret when ctrl f or ctrl h is pressed
give keyboard focus to dnspy seems to work most of the time closes
fix delegate signature
fix move caret left code
clear bp events when closed
remove prop from iface
make sure version string isn t too long
remove trailing _ from opcode in url
modify all exceptions at once
move check inside the lock
store decl type in a local
don t add language sub dirs that only contain vs mef localized files
set new version
add module bp remove
update scrollviewportverticallybypage to move last line to top
fix hex caret pos
save current brace pair before refreshing all tags
convert to struct
update assembly node tooltip colors
verify that source is this
make sure that the compile button isn t disabled after compilation has finished
update asm resolver
clear dict when disposing menu item ctx
update default hex offset bit size calculation
merge pull request from azazkamaz master
move search tool window to bottom location by default
update beta version
raise layoutchanged after bufferlineschanged
update color priority
if first frame is null keep looking until one with a non null function is found
update pe checkers
merge pull request from slowlogicboy sortassembliesctxmenu
fix setting bit
make tabbutton internal
don t create breakpoints in abstract methods
trace msg allow more escape sequences
add dir len check
catch exception in case it s an invalid filename
don t set isunhandledexception to true if it s a unity game
use loop var
better error handling
reenable all hotkeys once file menu closes resolves
fix group box was not translatable
typo and link fix
remove unecessary converts
rather than yellow highlight just make the node highlightling the same whether the contol has focus or not
ensure that quick connect items always have a panel set even if the con default panel setting is empty
load the default domain when opening the form
bumped patch version
only delete reg key value if the value exists
startup clean up some using directives and sort them
wired up the logging level selection
fix build error cleanup
fix release build failure
remove unnecessary convert
resolved bug where rearranging nodes by drag n drop would not save changes
fix reconnect to previously opened sessions on startup
react to icon change events
fix bug where we could not turn custom password encryption off
addl fix to ensure loading correct settings file
safer cast and value access when creating float window
fix bug where window text wasnt updated when choosing to create new connection from dialog popup
fix npe when importing from the connection tree
dispaly wait cursor when options are loading
fix issue with undeclared variable
add log message
remove a call that was causing many unnecessary host pings should help
fix problem saving confcons xml
add logic to the xmlconnectionsdocumentcompiler to respect the fullfileencryption flag
add a summary msg to one of the native method constants
set window scaling to dpi
saved lowercase searchtext to variable and used it for the comparison
resolved build failure
add missing translation
add cefsharp scheme to be able to display local html files e g the readthedocs documentation
add note regarding a minor efficiency bug to investigate
better fix for settingspath location
check for nulls
give focus to frmmain when clicking on it fixes
test for confversion serialized
minor fixes cleanup
renamed function to be a bit more accurate
not used in portable version
credentials save when the credential manager raises its changed event
fix non portable build
fix error in test project
merge pull request from seantmill feature trackintree
copy hostname on folder causes exception
set password box text masking
fix complier warning
fix null pointer reference due to missing parantheses
made the connection node name column take up all available space similar to previous implementation
only show putty provider if it has sessions
minor code cleanup
simplified some code
fix db schema error
minor code clean up
mremoteng import doesnt need to be static
changed background color to native powershell color
bumped assembly version
fix launching external tools
updated assembly info
fix supress warning
resolved bug where hitting the delete key would bring up the prompt to delete root nodes and putty nodes
remove dead code
add necessary call to save the csv file to disk
remove more vb referneces
saving settings before exiting and avoid early return
made a ui buttons text translatable
fix bug where we could not drop between items rearrange
we want only valid info
add null propagation check for single click handler resolves
changed the credential serializer lifecycle test to include password de encryption
fix title of options page was not translatable
improvement in protocol to tab focus
remove unnecessary property from xml connection loader
additional log adjustments
close is successful so at least we can reelase some resources for now
moved getconnectionicons call from frmmain to startup
updated sql upgrader class to reflect recent changes fixes
slight optimization of updating the tree when deleting a node refresh one row rather than the whole tree
add convenience ctor to the pagesequence class
add a few more tests for the legacyrijndaelcryptographyprovider
remove references to the my namespace
minor security page layout change
get protocol type descriptions by nameof instead of accessing them directly
password loaded when editing cred repo
hide credentialrecord in property grid if its inherited or if we re viewing the rootnode
suppress javascript errors
dispose calls close whitespace cleanup
resolved bug that broke the child node only ever has one parent rule
just a comment
remove unnecessary parens
remove unecessary converts
fix port scan window scaling
remove unused code
do not log to roaming appdata folder on default settings
fix some more empty string deserialization bugs
remove unused code
fix double click support for selecting cred repo type
remove test exception
copy hostname fix
move localization into the ctor
more code clean up
avoid null pointers when race conditions or faults are present in protocol closing
fix ugly unthemed border around connection search box
merge pull request from pfjason develop
fix an oversight with bool initialization
only access first array item if array has any items
made iconnectionsupdatechecker implement idisposable
remove redundant initialization
add test validating that an empty password cannot be passed to an encryptor
renamed inheritance copy to clone
resolved bug with the reconnect context menu item being permanently disabled for containers
add some details to stacktrace messages
fix build errors
improved display of ngcheckbox on hidpi devices
fix copy paste error
fix encryptor class
fix test failures
made sqlbulkcopy column position independant
remove more unnecessary code
resolved small bug with uninitialized sql dbs
remove inline comment per style feedback
close splashscreen when error occurs while loading connections
fix bug with starting ext tool on folder when nothing selected
xmlconnectionsdeserializer now implements ideserializer
layoutsettingsloader fix variable name for non portable build
fix failing tests
add test to verify
always set file menu position to the top left
minor code cleanup
fix exception opening in designer
add required property to the icryptographyprovider interface to get the underlying crypto algorithm
fix minor bug in the test value ranomizer
avoid exception when disposing external tools items
close the splash screen if it exists when an unhandled exception occurs
re add options menu icon
scale the putty image on the advanced page
add extra test to cover setting the authenticator max attempts
we can now edit cred repos
culture independent parsing of the configuration version
disabled full row select to be more in line with previous functionality
quick fix for
do nothing if double click on close is not set
remove dead code
add check to prevent dragging a node onto the folder in which it already resides
add mremoteng logo to splash screen
resolved bug where some new connections containers would not respect the default connection info and default inheritance settings
re remove waitforinputidle
minor clean up
fix bug in connection tree context menu
putty nodes and root putty node should not have context menu items import or export
update message change unhandled messages to false
right clicking on an empty connection panel no longer displays the context menu
jenkins doesn t like this
new folders created with new folder as the text
fix case sensitivity when searching for nodes
moved the iauthenticator into the security authentication namespace
add a beginedit at the end of the addnode function
made startup a singleton
changed the aeadcryptographyprovider to use the localized error message when decryption fails
bumped patch version
correct file name
fix cred id showing up in config window
update datatabledeserializer cs
slightly improved the setup of runtime isportableedition
resolved issue when clicking in connection tree white space
add some artificial delay to a test to help ensure it doesnt fail due to completing too quickly
fix a brittle and currently failing test
screenshot method change
do not automatically enable inheritance on connections imported from ad fixes
double include switch to handler for doubleclick yet
fix a cast
component check now tests for the correct rdp version
resolved exception when closing a connection group top level dps tab
fix complier warning
prevent exception at startup
prevent log file writing when option is not set
fix ssh transfer window scaling
handle events only when theming is active
remove unecessary convert
fix enhancedmode value not being translated
moved program initialization from frmmain to programroot
esc key now cancels options window changes
we can now open the log file from the options page
remove old code for serializing the complete xml file
add exception catching to a customaction method
add func for calculating scaled width
resolve logging bug where cmdline args arent shown
resolved bug where some context menu hot keys would not work after opening the menu on the root node
log exception messages only
add a method to the iconnectiontree interface to make testing another class possible
fix the http cef render engine not being localised properly
add double click support for toggling tree node expansion existed in previous version this was a feature regression
missed a commit for the copy clone rename
scale search image size
remove unecessary convert
add test for multi folder recursive inheritance
option page weren t displaying target language so i add apply language in frmoptions
remove all destructors since they were a very bad idea to implement
extracted method to improve code readability
chnaged splash screen form to display correctly on a high dpi device
save supported by the remoteconnectionssyncronizer
refactoring extracted a method in frmoptions that sets the initially selected page
fix issue with using the proper tree images for the putty nodes
minor code cleanup
ensure putty root node returns correct tree node type related to
activetheme is only saved in setting and applied on restart
fix bug that occurs on startup when non english language is selected
made search match nodes highlighted in yellow
ensured that the rootnodeinfo is not null during decryption
remove unused code
use root node when creating new connection folder and no node is selected
fix bug that would occasionally cause crashes when clicking
corrected order of connection info favorite property
remove unnecessary code
bumped assembly version for next pre release
resolved small bug in the xml cred repo editor page validation
fix minor complier warning
deregister vnc event handler on disconnect
add some more code around loading converted harvested cred repo
fix null ref error when viewing folders in some situations
improved fips policy check log output
renamed variable to be more in line with other uses in the class
add null guard when loading default conn info properties
more code clean up
more code cleanup
small cleanup in startup cs
tracks the active tab in the treeview
aesgcm implements icryptographyprovider
add custom type conversion support to the saveto method of defaultconnectioninfo
remove unnecessary instance var
created test to ensure decryption failures throw an exception in aeadcryptographyprovider
fix nullreferenceexcepction in renaming node
fix build failure
fix nullref error when attempting to delete a node when nothing is selected
fix possible but highly unlikely null assignment
remove old code
resolved bug when clicking in blank space in the connection tree
simplified save filter a bit
remove duplicate property after merge
xml cred deserializer now ensures empty strings are used for any missing xml attributes
fix bug where app would crash when no tree node selected and trying to import from file
putty now uses credential record
undo a line that was add for testing sorry
fix a variable that should have been renamed
fix crash when add node with no treenode selected
fix one case where visible connection tree width wasnt being updated correctly
cred manager no longer groups items
fix bug where our custom drag sink was being overwritten
fix a boo boo
remove more unused code
remove unnecesary convert
made the nonce size a bit smaller for ccm mode to avoid some issues with large nonces
resolved bug with settings file upgrade overwriting new data
whitepsace clean up
fix protection level non portable build failure
fix serialization bug
fix a typo in the connectiontreemodel s notifypropertychanged event propogation
fix bug where selected row would be deselected after committing a change to the list
fix copyfrom bug
forgot the idisposable
fix minor bug with passwordauthenticator
minor formatting fix
one more minor rename
fix exception that would occur if the repo list xml file did not exist
fix issue where resetpanels did nothing
renamed sessionchangedeventhandler to puttysessionchangedeventhandler
add a defaulttext mode to the ng search box
add null test for the composite click handler
don t attempt to connect to folders
fix error where connecting to vm on localhost
made a few save calls async these probably were async before and were accidentally changed
xml credential serializer now serializes domain value
fix test name
opening a loading a putty session and showing the putty configuration window will no longer freeze mremoteng
bump assembly version
promoted cred file path to a public property
renamed function to be a bit more descriptive
created more test cases for the externaltoolargumentparser
fix largeaddressaware test
fix bug with calling collectionchanged when sorting container children
extracted method for initializing the root tree node
resolved bug where export flag would not get set on the exported connections file
slight optimization to prevent window flicker when deleting a folder with many connections
domain is now available on connections with protocol int app
context menu item enable disabled based on node type is now working again
fix log message for debugging exception
slight optimization of the treelistview canexpandgetter
encrypt correct db field
resolved issue with unit test we could not select a tree object without that object being visible
commented out unused variable
fix minor bug with add a duplicate default session in some situations
last fix to ensure loading correct settings file
treelistview objects now update immediately when their openconnections list contents change
use the task dialog for confirming node deletions esc key now closes dialog
prevent exception when remove connection from list
putty config file path was incorrectly parsed
add get set property keyderivationiterations to the icryptographyprovider interface
resolved broken build references to drag events
if false is confusing no need for it if no else then
moved config window localization into the ctor
fix bug where user domain pass inheritance wasnt being shown
fix bug where any sessions with spaces in the name would be filtered out
set protocol type when importing from ad possible fix for
remove dependency to microsoft visualbasic in import cs
remove issimpledropsink restores item sorting for
increment version number
overrode the equals method for connectioninfo equality based on constantid
jump between octet fields with tab fixes
fix virtual call in constructor
add test to ensure that strings with international characters are serialized and deserialized correctly
cleaned up portscan cs
speed up theme option page load
fix issue with creating node hierarchy
ability to run external tools on folders close
