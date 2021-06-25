speed up todo
fix null pointer exception in pshader
remove redundant static modifier
compare stack frames by method that is invoked this way we more correctly remember variables expansion state
assert renew memtable is only used when durable writes false
add missing return in the example
remove redundant no arg constructor
fix audiotrack setplaybackrate error handling
remove todo as it works
fix documentation for wifi scanresult timestamp
add support for using a transaction to perform batch tasks
making sure that there is no notification when focus changes from something to nothing
add deprecated annotation to util getsize
add status code javadoc
remove extraneous comments
check mwebviewcore to prevent potential npe
reset last rendered text on render end
websockets now need this method
remove nullable from getvirtualfile since it is almost never null
set activity state correctly when ending activity
remove dead import
fix soft dependencies broken by previous commit
make nodeidx public
dialog should update status if apply is pressed
add a semicolon to stop qdox s whining
delete help button from postfix completion configurable
fix the droidgap activity lifecycle broken issue
fix null in help output
made evaluator constructor public to allow custom implementations
add pbap service start
remove checked exception from buffer readbytearray
remove blank line
add use idea classpath
remove test files after completion
add alternative method to create archive fix typo
smart expand api for structure view trees
do not ignore end of line symbols in bookmarks
fix annotation of getbehavior
add a comment
fix memory leak in shutdown of networksystemimpl
switch back the divider to true so the behavior keeps the same for people
add notification for a disabled stream due to excessive runtime
fix compiler warning
remove unused method
fix backgroundattribute bug
avoid compiler warnings
awaken listview scroll bars for overscroll
spell checker fixes narrowed scope made less visible where possible
temporarily ignore failing new test
fix the titleparsertest to reflect recent updates in behavior
fix eclipse javadoc warning
don t expand the searchview if already expanded
fix recent regression in drawablecontainer
remove extra blank line
project thumbnail should have the same aspect ratio
enable new building model
don t hold psi in local search scope of ssbasedinspection
remove redundant call v done in closurecodegen
make git status command silent
ignore incomplete tests
exclude nofrom test from hsqldb
make blocksuppressalstatus constructor public so that it can be constructed by the provider
add missing break
add default ctor to lexer
comment failing test
made ghserver extendable
fix source formatter
remove debug message that claims to be a error message
extending async testunit timeout
disable reloading for scripts with dates in the future
remove obsolete comment
add todo mid term goal is to implement nice functional immutable
fix setter in runtimeenvironment lucene locking
allow null values for json in collectionmetainfo
fix initial pull not displaying any text
add parsing of cython typedefs
remove username attribute from webattributes class
add missing final keyword
fix broken test
revert clear zombies after they are remove from timer list
remove debug line
fix loading theme color for non first webview load
fix npe in evaluate handler
engines should have a block name
remove extra import in businesssubscriptiontransitionrecorder
apparently firebird does not support ddl in transactions
add an assert to rowdataresolver getdata
reduced too broad catch block
add decent message when cusom matcher is used
ignore bad unit test
supressing rawtypes warning in test class
disabling all version removal for test
remove unused field
fix mappedrecordcursor bug
add getter for genericvertex max index
remove id keyword from all remaining files in tomcat
use the same stop
remove type arguments as compiler will infer them
remove unnecessary suppresswarnings
end handle disappears when the selection is short
remove unnecessary import in displayutils
reset joda time to use system clock after test
reset utterance counter upon stop
add noop unregisterreceiver to complement existing noop registerreceiver
update uploads view when finish upload is deleted
fix max stacksize on potions
enable optimized readers by default
fix possible problem with closed db
messageoutput now has getname
make msessioncallback null in tvview unbindtvinput
refactor find state and scrolling
add missed initializer for setter
remove print statement which was probably set during debugging
add marker for code improvement w r t ircapi instance
add deprecated annotations to declarations with deprecated javadoc tags
do not return subscriber connection to prevent getting events
add tostring to torodocument
add getspawnlocation to the world api
remove empty line
granular navigation uses mcontentdescription instead of getcpontentdescription s
add gl2_source_radio_input to hidden message fields
break loop on finding the first non printable character
deleting unused field
fix download complete message button
fix miscalculation of grid section bottom most edge
add missing uri import
fix bug in music when stop is called
remove typo in lwjglcanvas preventing the engine to build
debug logging remove
remove an accidental line break to retain the standard code formatting
remove spurious break
expanding properties fix eg fullname
suppress unchecked warning for read method
fix sendnotifications called twice
recycle the comet flag pretty large glitch
fix bug with local index not showing after switching tabs
fix check for same class in artifactcacheconnectevent
fix the search header on search results
fix copy pasted comment
make cfinfo a static inner class
remove pointless javadoc block
incorporated pr comment
reset when release
don t send error in readyforquerylistener
help id corrected
remove hardcoded position for scrolltoposition
expose enum argument method
remove an obsolete comment
don t escape html twice
add smalltest for continuous tests
avoid massive logging
fix npe when provider returns null cursor
allow the global naming context to be reset
remove obe comment
fix emergency call when in launch transition
drop cached qualified name after changing nodes in subtree
setnewdata add resetlastposition
fix test data restore incorrectly deleted import
remove unnecessary call to recycle
remove unusage service type
fix the code style
add a exception log for javassistproxyfactory static initiaiization
call oddatasetchanged in listremoteviewsfactory constructor to guarantee that task list is populated
addressing review comments
fix error in javadoc sample
fix forgotten delegation
fix yaml language display name
deprecate also implementation method
fix year picker initial range
textfield s cursor doesn t blink in gwt backend
allow testing element icons author section remove
implemented pre key lookup
whoops add a line separator
make viewholder less visible static
makes params final
remove stray import
disable one of the externaljavacintegrationtest test cases
fixl add listener to avd combo
make apersistentvector subvector public
add invalid_suppress to lint checks so it will be reported at the correct level
don t cache source formatting for now
restores the scroll bar s of the web browser in the new version is available dialog on windows
avoid multiple calls to writeinternal
add compound visitor for dynamic type builder
make mesh static by default
add missing class
updated startup service so it compiles
porting fix to allow for a forward include to call getattributenames on the request in a sandbox
fix stupid copy paste typo
add testsuitesanitytestcase to check testsuite environment
fix off by one issue when remove from iterator
fix failed tests
really leave text selection mode when using the trackball
add the return for the save method in pdf
fix checkstyle again
remove unnused variable
kill activityview s background thread when it s destroyed
remove temporary test
should not allow setting of the world
add average current property
don t re throw exception in onthrowable
made pgllistener constructor public
remove visiblefortesting for matchmode
fix swapped lac cid in card inflater
with the new features some of the delombok based unit tests no longer work
fix a findbugs warning
sqlutils surpport odps
add method to return raw message object
i did not fix activity check correctly last time
don t force reload of ide for project options
rethrowexception wraps throwable s in undeclaredthrowableexception
disable other test
fix database upgrade
add missing test flag
adjust performance expectations
do not merge tweak overscroll
i am using the inline editor github and i forgot a type before info gettypename
fix broken ut
add default value submitted by bill barker
fix xposed not enabled
remove unnecessary imported package
add missing branch
prevent list size toggle after first selection
make accounttestsuite abstract
remove empty string
do not stop editing on ismodified invokation
add newline to nooutputexception
add universalenglish as new language to grammaticalrelation such that original sd and ud can be both used
set animation to null after canceling current anim
add csiro au as tld since is not listed on mozilla psl
don t remove all queued items if token null
allow null filenamepattern in requestlogconfiguration
remove potentially misleading exception message
disabling still broken test
fix bug where groups were being closed twice
openssl does implement ssl_ck_rc2_128_cbc_export40_with_md5
fix a copy paste misprint
add a call to super setspatial in setspatial
fix tests by running super teardown
increasing visibility for ben
updated tsviterator and created an asrelations processor
togglebutton could keep a reference to a previous drawable
fix checkstyle issue
cursortobulkcursoradapter close must call mcursor close instead of mcursor deactivate
print version message at server definition
ease the validations on databaseconfiguration url
fix copy n paste error
fix an serializable bug
fix typo in method isrotate d supported don t attempt to rotate images of unknown format
make clean compactions cleanup the row cache
add private constructor to threadsafelazyloadedivorytower
remove todo in telecommanager
fix action bar items missing when opening new message
made commandlineoptions getoptionsiterator package private
if null check version
remove bogus comment
remove unused import
add note on probable bug
fix option parsing when piping data in via stdin
fix minor eclipse compile warning
add isdebug method
fix a typo in a test method name
suppress incorrect warning
add empty line
make predicate package local
recreate overlay view after the tis crashes
numberpicker truncates the edited value
align mockito after with mockito timeout to take a long instead of int
try and handle death on startup a bit cleaner
set log level to trace for package action support broadcast in the recoverypercolatortests testsinglepercolator_recovery test
don t force actionmenuitemview to use textallcaps
remove unused import
remove redundant annotation
fix bug where sslhandler returned wrong value for getclosenotifytimeoutmillis
fix logic error
add getcollection class to interface
fix on off option update error
disable caching explicitly to get rid of useless log entry
return the document if a stream is received
placed the completed call under the right braces
remove non functional history tree provider
invalidate the cached ripple drawable mask if mask layer changes
allow external file picker
make sure chat list updates after paging
remove not needed call of bytebuffer clear
mark getconnection that takes a username password as deprecated
add tostring to golighttype
fix broken build
fix supervisorproperty display name
include correct redirection messages in the har list returned by the api
fix wrong class filter condition
improve doc string
add change user password
changed power request to activation energy
make pyunbound local variable inspection enabled by default
remove unused line in official plugins test
add final modifier to atomicboolean isstarted
revert previous fix
ok so we need this
remove unneeded method
fix wrong states when waking up from doze via touch
guarding moved pass checkregexp with skipnontranspilationpasses flag
missed required type id for a junit
update the footer indicator paint when footer color is set programmatically
remove unnecessary unused suppression
remove stray semi colon
fix build after revert
fix npe when branding isn t present
add suppresswarnings serial to an exception
give focus to text input in addtask activity
fix crash when scrolling a layer
fix small rendering bug when item is hovering over advancedslot
voicemail number setting is throwing exception
fix useless logging
if there is no error do change the target
add null check in viewoverlay
dismiss system windows when search is launched
add a ms tag to the debug time system out
fix usages may break code if conflicts were ignored
add a help option for the tests
remove component annotation
add a javadoc pointer from discretedomain to discretedomains
setting backround color if image comes directly from cache
remove requestlayout call since it s run via dynamicheighttextview
flip emergency_assistance_enabled to true
fix trivial equals implementation
allow override of calculated size
fix race condition
change intsetiterator to public
don t show sidebar menu by default
fix flag value collision
fix broken update
reenable a test
remove extraneous whitespace from pull request
fix erroneous conversion to radians
make eventual assertion in partitionservicesafetychecktest
fix null pointer check in sendextracommand
implemented getgl method in plwjgl but lwjgl not tested after latest
invalidate webview after title change
fixes caching plugin components reported by marin dzhigarov
fix a bug that could cause improper detection of long press selection mode in webview after a scale gesture
make class non public
remove superfluous suppresswarnings unused from putallpartitionawareoperationfactory
still return a package suspsended dialog if there is no owner
evaluate the injected javascript
don t performsend on onsave
remove unused enum value
set mpositionbeforescroll to page we notify switched to
committing a memory leak fix from michael koch making sure that players are properly released after a call
remove local that was masking the superclass field
add context accessor for subclasses
close the message logger so that the file is closed on windows
fix compile error caused by final keyword
fix a failing unit test
fix test compilation
add support to get right dialog from activity
suppress unchecked warning
enable cleaner hack so it consumes less disk space
remove testing code
keep this getter for consistency with localtachyoncluster
remove spurious blank line
add default case
remove completed fixme
break loop fast when one of connection seems to broken
fix splash progress disappearing on image repaint
fix for rajawalirenderer setcamera didn t update the
assert statement formatting
remove spurious system out println
remove unnecessary return type for method
remove todo for bad idea
